// ============================================================
// WeatherAiDotNet – Local RAG (Retrieval-Augmented Generation) Demo
// ============================================================
// This is the application entry point.  It orchestrates the
// full RAG pipeline:
//
//   1. Validate paths and optional external tools (OCR CLI).
//   2. Probe the embedding model and generation model.
//   3. Recursively scan a folder for PDF files.
//   4. For each PDF:
//        a. Compute a lightweight fingerprint (size + last-write time).
//        b. Skip the file if the fingerprint matches what is already
//           stored in the SQLite database (incremental indexing).
//        c. Extract page text, split into overlapping chunks, and
//           embed each chunk.
//        d. Optionally extract embedded images, OCR them, and embed
//           the resulting text.
//        e. Atomically store all vectors for the file in SQLite.
//   5. Enter an interactive Q&A loop where:
//        a. The user's question is embedded.
//        b. The most relevant chunks are retrieved from the database
//           using hybrid (semantic + lexical) search.
//        c. The retrieved chunks are assembled into a prompt context
//           and sent to the local generation model for an answer.
// ============================================================

using Serilog;
using Serilog.Extensions.Logging;
using System.Text.RegularExpressions;
using WeatherAiDotNet.Configuration;
using WeatherAiDotNet.RagIndexing;
using WeatherAiDotNet.Services;

// Configure Serilog as the single logging pipeline for the application.
// Logs are written to persistent rolling files under the repository "logs" folder.
// This logger is also bridged into Microsoft.Extensions.Logging so libraries
// that accept ILogger can emit through the same sink configuration.
var logsPath = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "logs", "weather-ai-.log"));
Directory.CreateDirectory(Path.GetDirectoryName(logsPath)!);

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Information()
    .Enrich.FromLogContext()
    .WriteTo.File(
        logsPath,
        rollingInterval: RollingInterval.Day,
        retainedFileCountLimit: 14,
        shared: true)
    .CreateLogger();

using var loggerFactory = new SerilogLoggerFactory(Log.Logger, dispose: false);
// Configure logging for the native llama.cpp layer first — before either service
// calls LlamaNativeService.Configure() — so the callback is registered before
// the native library loads.
LlamaNativeService.ConfigureLogging(loggerFactory);
EmbeddingService.ConfigureLogging(loggerFactory);
LlamaGenerationService.ConfigureLogging(loggerFactory);

try
{
    // Parse command-line arguments and apply defaults for any unspecified options.
    var appOptions = AppOptions.FromArgs(args);

    // effectiveOcrCliPath may be cleared below if the OCR binary fails validation.
    var effectiveOcrCliPath = appOptions.OcrCliPath;

    // --- Pre-flight checks -------------------------------------------------------

    if (!Directory.Exists(appOptions.PdfFolderPath))
    {
        Log.Error("PDF folder not found: {PdfFolderPath}", appOptions.PdfFolderPath);
        return;
    }

    if (!File.Exists(appOptions.ModelPath))
    {
        Log.Error("GGUF model not found: {ModelPath}", appOptions.ModelPath);
        return;
    }

    if (!File.Exists(appOptions.EmbeddingModelPath))
    {
        Log.Error("Embedding GGUF model not found: {EmbeddingModelPath}", appOptions.EmbeddingModelPath);
        return;
    }

    // Print hardware/backend information so the user knows how the model will run.
    Log.Information("Using LLamaSharp with {LlamaBackend} backend preference on this Intel Arc 140V + Intel AI Boost system.", appOptions.LlamaBackend);
    Log.Information(
        "LLamaSharp tuning: gpu-layers={GpuLayers}, threads={Threads}, batch-threads={BatchThreads}, batch-size={BatchSize}, ubatch-size={UBatchSize}.",
        appOptions.GpuLayers,
        appOptions.Threads,
        appOptions.BatchThreads,
        appOptions.BatchSize,
        appOptions.UBatchSize);

    Log.Information(
        "Retrieval tuning: top-k={TopK}, retrieval-pool={RetrievalPool}, chunk-size={ChunkSize}, chunk-overlap={ChunkOverlap}.",
        appOptions.TopK,
        appOptions.RetrievalPool,
        appOptions.ChunkSize,
        appOptions.ChunkOverlap);
    Log.Information(
        "Generation sampling: temperature={Temperature}, top-p={TopP}.",
        appOptions.Temperature,
        appOptions.TopP);
    Log.Information("Initializing LLamaSharp...");

    // Validate the optional OCR CLI (e.g., tesseract.exe) before the indexing loop.
    // If it can't be launched we disable image OCR for this run.
    if (!string.IsNullOrWhiteSpace(appOptions.OcrCliPath))
    {
        var ocrCheck = await CliToolService.ValidateAsync(appOptions.OcrCliPath, "--version");
        if (!ocrCheck.Success)
        {
            Log.Warning("OCR executable could not be validated and will be ignored: {OcrCliPath}", appOptions.OcrCliPath);
            Log.Warning("OCR validation error: {Message}", ocrCheck.Message);
            effectiveOcrCliPath = string.Empty;
        }
    }

    // Create the image output directory now so we don't fail mid-ingest.
    if (appOptions.IncludeImages)
    {
        Directory.CreateDirectory(appOptions.ImagesOutputPath);
    }

    // --- Discover and index PDF files --------------------------------------------

    // Ensure the SQLite database file and tables exist before we start writing.
    VectorStoreService.EnsureDatabase(appOptions.DbPath);

    Log.Information("Preparing embedding model...");

    // --- Probe the embedding model ------------------------------------------------
    // We send a short test string through the model to confirm it loads and produces
    // a non-empty vector.  If this probe fails, we fall back to hash-based embeddings
    // for the entire run.

    var embeddingProbe = await EmbeddingService.ProbeAsync(
        appOptions.EmbeddingModelPath,
        appOptions.LlamaBackend,
        appOptions.PreferGpu,
        appOptions.GpuLayers,
        appOptions.ContextSize,
        appOptions.Threads,
        appOptions.BatchThreads,
        appOptions.BatchSize,
        appOptions.UBatchSize);

    // If the preferred backend probe fails, run a conservative CPU-only probe to
    // determine whether the model itself works but current runtime settings are too
    // aggressive for embedding inference on this machine.
    if (embeddingProbe.Vector is null)
    {
        var conservativeEmbeddingProbe = await EmbeddingService.ProbeAsync(
            appOptions.EmbeddingModelPath,
            backend: "cpu",
            preferGpu: false,
            gpuLayers: 0,
            contextSize: Math.Min(appOptions.ContextSize, 2048),
            threads: Math.Max(1, Math.Min(appOptions.Threads, 8)),
            batchThreads: Math.Max(1, Math.Min(appOptions.BatchThreads, 8)),
            batchSize: Math.Min(appOptions.BatchSize, 64),
            uBatchSize: Math.Min(appOptions.UBatchSize, 64));

        if (conservativeEmbeddingProbe.Vector is { Length: > 0 })
        {
            embeddingProbe = conservativeEmbeddingProbe;
            Log.Warning("Primary embedding probe failed, but conservative CPU embedding probe succeeded. Using model embeddings with conservative settings.");
        }
    }

    // --- Probe the generation model -----------------------------------------------
    // Similarly, we try to load the generation model weights before the Q&A loop so
    // a missing or corrupt GGUF is caught early.

    var generationProbe = await LlamaGenerationService.ProbeAsync(
        appOptions.ModelPath,
        appOptions.LlamaBackend,
        appOptions.PreferGpu,
        appOptions.GpuLayers,
        appOptions.ContextSize,
        appOptions.Threads,
        appOptions.BatchThreads,
        appOptions.BatchSize,
        appOptions.UBatchSize);

    // Decide whether to use the real embedding model or the hash fallback.
    var useModelEmbeddings = embeddingProbe.Vector is { Length: > 0 };
    if (!useModelEmbeddings)
    {
        Log.Warning("Embedding runtime unavailable for LLamaSharp, falling back to local CPU hash embeddings.");
        if (!string.IsNullOrWhiteSpace(embeddingProbe.Diagnostic))
        {
            Log.Warning("Embedding diagnostic: {Diagnostic}", embeddingProbe.Diagnostic);
        }
    }
    else
    {
        Log.Information("Embedding runtime: {RuntimeBackend}.", EmbeddingService.GetRuntimeBackend());
        if (!string.IsNullOrWhiteSpace(embeddingProbe.Diagnostic))
        {
            Log.Information("Embedding initialization diagnostic: {Diagnostic}", embeddingProbe.Diagnostic);
        }
    }

    // Generation model is required; exit if it could not be loaded.
    if (generationProbe.Success)
    {
        Log.Information("Generation runtime: {RuntimeBackend}.", LlamaGenerationService.GetRuntimeBackend());
        if (!string.IsNullOrWhiteSpace(generationProbe.Message))
        {
            Log.Information("Generation initialization diagnostic: {Diagnostic}", generationProbe.Message);
        }
    }
    else
    {
        Log.Error("Generation runtime unavailable: {Error}", generationProbe.Message);
        return;
    }

    // Note for users on Intel AI Boost hardware: the NPU is not used by this stack.
    Log.Information("Intel AI Boost NPU detected, but the current LLamaSharp/llama.cpp stack in this project can use Vulkan GPU acceleration, not the NPU.");

    // The actual embedding dimension depends on the model.  Use the probe vector's
    // length if available; otherwise use the configured fallback size.
    var effectiveEmbeddingSize = useModelEmbeddings ? embeddingProbe.Vector!.Length : appOptions.EmbeddingSize;

    // Build a version stamp that captures embedding-relevant settings.
    // We append this stamp to each document fingerprint so unchanged PDFs are
    // still re-indexed automatically when the embedding model (or its effective
    // vector shape/configuration) changes.
    var embeddingModelInfo = new FileInfo(appOptions.EmbeddingModelPath);
    var embeddingVersionStamp = string.Join('|',
        Path.GetFileName(appOptions.EmbeddingModelPath),
        embeddingModelInfo.Length,
        embeddingModelInfo.LastWriteTimeUtc.Ticks,
        effectiveEmbeddingSize,
        appOptions.ContextSize,
        appOptions.BatchSize,
        appOptions.UBatchSize,
        appOptions.ChunkSize,
        appOptions.ChunkOverlap);

    // Startup visibility: this stamp is part of every stored fingerprint and is the
    // mechanism that forces safe automatic re-indexing when embedding-relevant inputs
    // change (model file, quantization swap, vector size, or chunk settings).
    Log.Information("Embedding index version stamp: {EmbeddingVersionStamp}", embeddingVersionStamp);

    // --- Incremental PDF indexing -------------------------------------------------

    var indexingSummary = await RagDatabaseIndexer.IndexAsync(
        appOptions.PdfFolderPath,
        appOptions.IncludeImages,
        PdfContentService.ReadPdfText,
        // Use chunking parameters from configuration so retrieval can be tuned for
        // fact-heavy PDFs without code changes. Smaller chunks usually improve hit
        // rate for direct lookup questions.
        text => PdfContentService.ChunkText(text, chunkSize: appOptions.ChunkSize, overlap: appOptions.ChunkOverlap),
        file => PdfContentService
            .ExtractPdfImageItems(file, appOptions.PdfFolderPath, appOptions.ImagesOutputPath, effectiveOcrCliPath)
            .Select(static item => new RagImageItem(item.IndexText))
            .ToList(),
        chunk => EmbeddingService.GetEmbeddingAsync(
            chunk,
            effectiveEmbeddingSize,
            useModelEmbeddings,
            appOptions.EmbeddingModelPath,
            appOptions.LlamaBackend,
            appOptions.PreferGpu,
            appOptions.GpuLayers,
            appOptions.ContextSize,
            appOptions.Threads,
            appOptions.BatchThreads,
            appOptions.BatchSize,
            appOptions.UBatchSize),
        (relativePdfPath, fingerprint) =>
        {
            // Include embedding version in the stored fingerprint so switching to a
            // different GGUF (for example Q4_K_M), replacing a file in-place, or
            // changing key embedding/chunk parameters forces a safe re-index.
            var versionedFingerprint = $"{fingerprint}|embed={embeddingVersionStamp}";
            return VectorStoreService.IsDocumentUpToDate(appOptions.DbPath, appOptions.CollectionName, relativePdfPath, versionedFingerprint);
        },
        (relativePdfPath, fingerprint, vectors) =>
        {
            var versionedFingerprint = $"{fingerprint}|embed={embeddingVersionStamp}";
            var storedVectors = vectors
                .Select(v => new WeatherAiDotNet.Models.StoredVector(appOptions.CollectionName, relativePdfPath, v.ChunkIndex, v.Text, v.Vector))
                .ToList();

            VectorStoreService.ReplaceDocumentVectors(
                appOptions.DbPath,
                appOptions.CollectionName,
                relativePdfPath,
                versionedFingerprint,
                storedVectors);
        },
        message => Log.Information("{Message}", message),
        (message, ex) => Log.Warning(ex, "{Message}", message));

    Log.Information(
        "Re-indexed {ReindexedDocuments} PDFs, skipped {SkippedDocuments} unchanged PDFs.",
        indexingSummary.ReindexedDocuments,
        indexingSummary.SkippedDocuments);

    if (indexingSummary.ReindexedDocuments > 0)
    {
        Log.Information("Indexed {TotalIndexedChunks} chunks.", indexingSummary.TotalIndexedChunks);
        if (appOptions.IncludeImages)
        {
            Log.Information("Processed {TotalImageItems} embedded image items.", indexingSummary.TotalImageItems);
        }
    }

    // Guard: refuse to enter the Q&A loop if there is nothing to search.
    if (!VectorStoreService.HasAnyVectors(appOptions.DbPath, appOptions.CollectionName))
    {
        Log.Warning("No text or image chunks are available in the vector database for collection {CollectionName}.", appOptions.CollectionName);
        return;
    }

    Log.Information("Ingestion complete.");
    Log.Information("Ask questions about the PDFs (press Enter on an empty line to exit):");

    // --- Interactive Q&A loop ----------------------------------------------------

    while (true)
    {
        Console.Write("Weather Inquiry:> ");
        var question = Console.ReadLine();
        Console.WriteLine("Thinking...");
        // An empty line signals the user wants to exit.
        if (string.IsNullOrWhiteSpace(question))
        {
            break;
        }

        // Embed the question using the same model/settings used during ingestion so
        // the question vector lives in the same semantic space as the chunk vectors.
        var questionVector = await EmbeddingService.GetEmbeddingAsync(
            question,
            effectiveEmbeddingSize,
            useModelEmbeddings,
            appOptions.EmbeddingModelPath,
            appOptions.LlamaBackend,
            appOptions.PreferGpu,
            appOptions.GpuLayers,
            appOptions.ContextSize,
            appOptions.Threads,
            appOptions.BatchThreads,
            appOptions.BatchSize,
            appOptions.UBatchSize);

        // Retrieve the most relevant chunks using hybrid semantic + lexical search.
        var matches = VectorStoreService.Search(
            appOptions.DbPath,
            appOptions.CollectionName,
            question,
            questionVector,
            appOptions.TopK,
            appOptions.RetrievalPool);

        if (matches.Count == 0)
        {
            Log.Information("No relevant context found.");
            continue;
        }

        // Concatenate retrieved chunks into a single context block.
        // The [N] prefix and Source tag help the model cite its sources.
        var context = string.Join("\n\n---\n\n", matches.Select((m, idx) => $"[{idx + 1}] (Source: {m.Source})\n{m.Text}"));

        // Two-step grounded answering flow:
        // 1) Extract evidence first (quoted/near-quoted facts with citations),
        // 2) Compose the final answer from that evidence only.
        // This reduces hallucinations and increases the chance that returned answers
        // are directly supported by PDF chunks.
        var evidencePrompt = $"""
            You are a retrieval verifier for a U.S. Air Force weather assistant.
            Using ONLY the numbered context snippets, extract the minimum evidence needed to answer the question.
            Rules:
            - Return 2 to 6 concise bullet points.
            - Each bullet MUST end with at least one citation like [1] or [2].
            - Do not include facts not present in the snippets.
            - If there is not enough evidence, return exactly: NO_EVIDENCE

            Context snippets:
            {context}

            Question: {question}

            Evidence:
            """;

        var extractedEvidence = await LlamaGenerationService.GenerateAnswerAsync(
            appOptions.ModelPath,
            evidencePrompt,
            maxTokens: 900,
            appOptions.LlamaBackend,
            appOptions.PreferGpu,
            appOptions.GpuLayers,
            appOptions.ContextSize,
            appOptions.Threads,
            appOptions.BatchThreads,
            appOptions.BatchSize,
            appOptions.UBatchSize,
            appOptions.Temperature,
            appOptions.TopP);

        var evidenceText = extractedEvidence.Trim();
        var hasEvidenceCitations = Regex.IsMatch(evidenceText, @"\[\d+\]");

        string answer;
        if (string.IsNullOrWhiteSpace(evidenceText)
            || evidenceText.Contains("NO_EVIDENCE", StringComparison.OrdinalIgnoreCase)
            || !hasEvidenceCitations)
        {
            // If evidence extraction is weak, return the grounded fallback early
            // instead of asking the model to speculate.
            Log.Information("Evidence extraction returned insufficient support. Returning grounded fallback response.");
            answer = "I don't know based on the indexed documents.";
        }
        else
        {
            var answerPrompt = $"""
                You are a helpful U.S. Air Force weather analyst assistant.
                Answer the question using ONLY the evidence bullets below.
                Write a clear, complete answer in full sentences.
                Preserve key terminology from the evidence when relevant.
                After each fact, include supporting citations, e.g. [1] or [2].
                If the evidence does not support an answer, say exactly:
                I don't know based on the indexed documents.

                Question: {question}

                Evidence bullets:
                {evidenceText}

                Answer:
                """;

            answer = await LlamaGenerationService.GenerateAnswerAsync(
                appOptions.ModelPath,
                answerPrompt,
                maxTokens: 2048,
                appOptions.LlamaBackend,
                appOptions.PreferGpu,
                appOptions.GpuLayers,
                appOptions.ContextSize,
                appOptions.Threads,
                appOptions.BatchThreads,
                appOptions.BatchSize,
                appOptions.UBatchSize,
                appOptions.Temperature,
                appOptions.TopP);

            // Enforce grounding: if the model returned only whitespace or citation
            // tokens with no surrounding text, replace with the safe fallback.
            var trimmed = answer.Trim();
            var isCitationOnly = !string.IsNullOrWhiteSpace(trimmed) &&
                                 Regex.IsMatch(trimmed, @"^\s*(\[\d+\]\s*[,;]?\s*)+$");

            if (string.IsNullOrWhiteSpace(trimmed) || isCitationOnly)
            {
                Log.Warning("Model response was empty or citation-only; returning grounded fallback response.");
                answer = "I don't know based on the indexed documents.";
            }
        }

        Log.Information("Answer: {Answer}", answer);
        Console.WriteLine();
        Console.WriteLine(answer);

        var citedReferenceNumbers = Regex.Matches(answer, @"\[(\d+)\]")
            .Select(match => int.TryParse(match.Groups[1].Value, out var number) ? number : -1)
            .Where(number => number >= 1 && number <= matches.Count)
            .Distinct()
            .OrderBy(number => number)
            .ToList();

        if (citedReferenceNumbers.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("References:");

            foreach (var referenceNumber in citedReferenceNumbers)
            {
                var source = matches[referenceNumber - 1].Source;
                Console.WriteLine($"[{referenceNumber}] {source}");
            }
        }

        Console.WriteLine();
    }
}
catch (Exception ex)
{
    Log.Fatal(ex, "Unhandled exception while running WeatherAiDotNet.");
}
finally
{
    Log.CloseAndFlush();
}
