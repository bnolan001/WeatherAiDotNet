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
using WeatherAiDotNet.Models;
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

    // --- Discover PDF files -------------------------------------------------------

    Log.Information("Reading PDFs...");
    var pdfFiles = Directory
        .EnumerateFiles(appOptions.PdfFolderPath, "*.pdf", SearchOption.AllDirectories)
        .OrderBy(x => x, StringComparer.OrdinalIgnoreCase)
        .ToList();

    if (pdfFiles.Count == 0)
    {
        Log.Warning("No PDF files found under: {PdfFolderPath}", appOptions.PdfFolderPath);
        return;
    }

    // Ensure the SQLite database file and tables exist before we start writing.
    VectorStoreService.EnsureDatabase(appOptions.DbPath);

    Log.Information("Found {PdfCount} PDF files.", pdfFiles.Count);
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

    // --- Incremental PDF indexing loop -------------------------------------------

    var reindexedDocuments = 0;
    var skippedDocuments = 0;
    var totalIndexedChunks = 0;
    var totalImageItems = 0;

    foreach (var file in pdfFiles)
    {
        // Use a path relative to the PDF root as the stable document identifier so
        // moving the root folder doesn't invalidate the database.
        var relativePdfPath = Path.GetRelativePath(appOptions.PdfFolderPath, file);

        // The fingerprint encodes file size and last-write time; it's fast to compute
        // and catches both content changes and silent re-saves.
        var fingerprint = BuildDocumentFingerprint(file);

        // If the stored fingerprint matches, the PDF hasn't changed — skip it.
        if (VectorStoreService.IsDocumentUpToDate(appOptions.DbPath, appOptions.CollectionName, relativePdfPath, fingerprint))
        {
            skippedDocuments++;
            Log.Information("Skipping unchanged PDF: {RelativePdfPath}", relativePdfPath);
            continue;
        }

        Log.Information("Indexing PDF: {RelativePdfPath}", relativePdfPath);

        // Extract plain text from every page of the PDF.
        string pdfText;
        try
        {
            pdfText = PdfContentService.ReadPdfText(file);
        }
        catch (Exception ex)
        {
            // Skip unreadable/corrupt PDFs rather than aborting the whole run.
            Log.Warning(ex, "Skipping unreadable PDF: {PdfPath}", file);
            continue;
        }

        var chunkIndex = 0;
        var fileVectors = new List<StoredVector>();

        // Split the full document text into overlapping word-window chunks.
        // chunkSize ≈ 1200 characters, overlap ≈ 200 characters keeps context
        // continuity across chunk boundaries.
        var chunks = PdfContentService.ChunkText(pdfText, chunkSize: 1200, overlap: 200).ToList();
        foreach (var chunk in chunks)
        {
            // Embed the chunk text to get its position in semantic vector space.
            var vector = await EmbeddingService.GetEmbeddingAsync(
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
                appOptions.UBatchSize);

            fileVectors.Add(new StoredVector(appOptions.CollectionName, relativePdfPath, chunkIndex++, chunk, vector));
        }

        // Optionally extract and index images embedded in the PDF.
        if (appOptions.IncludeImages)
        {
            var imageItems = PdfContentService.ExtractPdfImageItems(file, appOptions.PdfFolderPath, appOptions.ImagesOutputPath, effectiveOcrCliPath);
            totalImageItems += imageItems.Count;

            foreach (var imageItem in imageItems)
            {
                // Embed the image's OCR/metadata text so it can be retrieved just
                // like a regular text chunk.
                var vector = await EmbeddingService.GetEmbeddingAsync(
                    imageItem.IndexText,
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

                fileVectors.Add(new StoredVector(appOptions.CollectionName, relativePdfPath, chunkIndex++, imageItem.IndexText, vector));
            }
        }

        if (fileVectors.Count == 0)
        {
            // Nothing to store (empty PDF or no extractable content); move on.
            continue;
        }

        // Atomically delete the old vectors for this document and insert the new
        // ones along with the updated fingerprint.
        VectorStoreService.ReplaceDocumentVectors(
            appOptions.DbPath,
            appOptions.CollectionName,
            relativePdfPath,
            fingerprint,
            fileVectors);

        reindexedDocuments++;
        totalIndexedChunks += fileVectors.Count;
    }

    Log.Information("Re-indexed {ReindexedDocuments} PDFs, skipped {SkippedDocuments} unchanged PDFs.", reindexedDocuments, skippedDocuments);
    if (reindexedDocuments > 0)
    {
        Log.Information("Indexed {TotalIndexedChunks} chunks.", totalIndexedChunks);
        if (appOptions.IncludeImages)
        {
            Log.Information("Processed {TotalImageItems} embedded image items.", totalImageItems);
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
        Console.Write("> ");
        var question = Console.ReadLine();

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

        // Build a grounding prompt that instructs the model to answer naturally
        // while citing the numbered snippets inline. Keeping instructions concise
        // reduces the chance that the model echoes back only the citation numbers.
        var prompt = $"""
            You are a helpful U.S. Air Force weather analyst assistant.
            Answer the question below using ONLY the numbered context snippets provided.
            Write a clear, complete answer in full sentences.
            After each fact, add the snippet number in brackets, e.g. [1] or [2].
            If the snippets do not contain enough information to answer, say exactly:
            I don't know based on the indexed documents.

            Context snippets:
            {context}

            Question: {question}

            Answer:
            """;

        // Send the prompt to the local generation model and print the response.
        var answer = await LlamaGenerationService.GenerateAnswerAsync(
            appOptions.ModelPath,
            prompt,
            maxTokens: 400,
            appOptions.LlamaBackend,
            appOptions.PreferGpu,
            appOptions.GpuLayers,
            appOptions.ContextSize,
            appOptions.Threads,
            appOptions.BatchThreads,
            appOptions.BatchSize,
            appOptions.UBatchSize);

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

        Log.Information("Answer: {Answer}", answer);
        Console.WriteLine();
        Console.WriteLine(answer);
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

// --- Helper functions --------------------------------------------------------

/// <summary>
/// Builds a cheap fingerprint string for a file so we can detect changes without
/// reading the file's full content.  The fingerprint is "{bytes}:{ticks}" where
/// bytes is the file size and ticks is the UTC last-write time in .NET ticks.
/// This is fast but not cryptographically secure; it's sufficient for an
/// incremental-indexing skip check.
/// </summary>
static string BuildDocumentFingerprint(string filePath)
{
    var fileInfo = new FileInfo(filePath);
    return $"{fileInfo.Length}:{fileInfo.LastWriteTimeUtc.Ticks}";
}
