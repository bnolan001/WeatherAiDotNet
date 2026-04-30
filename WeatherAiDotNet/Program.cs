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

using WeatherAiDotNet.Configuration;
using WeatherAiDotNet.Models;
using WeatherAiDotNet.Services;

// Parse command-line arguments and apply defaults for any unspecified options.
var appOptions = AppOptions.FromArgs(args);

// effectiveOcrCliPath may be cleared below if the OCR binary fails validation.
var effectiveOcrCliPath = appOptions.OcrCliPath;

// --- Pre-flight checks -------------------------------------------------------

if (!Directory.Exists(appOptions.PdfFolderPath))
{
    Console.WriteLine($"PDF folder not found: {appOptions.PdfFolderPath}");
    return;
}

if (!File.Exists(appOptions.ModelPath))
{
    Console.WriteLine($"GGUF model not found: {appOptions.ModelPath}");
    return;
}

if (!File.Exists(appOptions.EmbeddingModelPath))
{
    Console.WriteLine($"Embedding GGUF model not found: {appOptions.EmbeddingModelPath}");
    return;
}

// Print hardware/backend information so the user knows how the model will run.
Console.WriteLine($"Using LLamaSharp with {appOptions.LlamaBackend} backend preference on this Intel Arc 140V + Intel AI Boost system.");
Console.WriteLine($"LLamaSharp tuning: gpu-layers={appOptions.GpuLayers}, threads={appOptions.Threads}, batch-threads={appOptions.BatchThreads}, batch-size={appOptions.BatchSize}, ubatch-size={appOptions.UBatchSize}.");
Console.WriteLine("Initializing LLamaSharp...");

// Validate the optional OCR CLI (e.g., tesseract.exe) before the indexing loop.
// If it can't be launched we disable image OCR for this run and warn in red.
if (!string.IsNullOrWhiteSpace(appOptions.OcrCliPath))
{
    var ocrCheck = await CliToolService.ValidateAsync(appOptions.OcrCliPath, "--version");
    if (!ocrCheck.Success)
    {
        var originalColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"OCR executable could not be validated and will be ignored: {appOptions.OcrCliPath}");
        Console.WriteLine(ocrCheck.Message);
        Console.ForegroundColor = originalColor;
        effectiveOcrCliPath = string.Empty;
    }
}

// Create the image output directory now so we don't fail mid-ingest.
if (appOptions.IncludeImages)
{
    Directory.CreateDirectory(appOptions.ImagesOutputPath);
}

// --- Discover PDF files -------------------------------------------------------

Console.WriteLine("Reading PDFs...");
var pdfFiles = Directory
    .EnumerateFiles(appOptions.PdfFolderPath, "*.pdf", SearchOption.AllDirectories)
    .OrderBy(x => x, StringComparer.OrdinalIgnoreCase)
    .ToList();

if (pdfFiles.Count == 0)
{
    Console.WriteLine($"No PDF files found under: {appOptions.PdfFolderPath}");
    return;
}

// Ensure the SQLite database file and tables exist before we start writing.
VectorStoreService.EnsureDatabase(appOptions.DbPath);

Console.WriteLine($"Found {pdfFiles.Count} PDF files.");
Console.WriteLine("Preparing embedding model...");

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
    Console.WriteLine("Embedding runtime: unavailable for LLamaSharp, falling back to local CPU hash embeddings.");
    Console.WriteLine("Could not generate embeddings from LLamaSharp. Using hash embeddings for this run.");
    if (!string.IsNullOrWhiteSpace(embeddingProbe.Diagnostic))
    {
        Console.WriteLine($"Embedding diagnostic: {embeddingProbe.Diagnostic}");
    }
}
else
{
    Console.WriteLine($"Embedding runtime: {EmbeddingService.GetRuntimeBackend()}.");
    if (!string.IsNullOrWhiteSpace(embeddingProbe.Diagnostic))
    {
        Console.WriteLine(embeddingProbe.Diagnostic);
    }
}

// Generation model is required; exit if it could not be loaded.
if (generationProbe.Success)
{
    Console.WriteLine($"Generation runtime: {LlamaGenerationService.GetRuntimeBackend()}.");
    if (!string.IsNullOrWhiteSpace(generationProbe.Message))
    {
        Console.WriteLine(generationProbe.Message);
    }
}
else
{
    Console.WriteLine("Generation runtime: unavailable.");
    Console.WriteLine(generationProbe.Message);
    return;
}

// Note for users on Intel AI Boost hardware: the NPU is not used by this stack.
Console.WriteLine("Intel AI Boost NPU detected, but the current LLamaSharp/llama.cpp stack in this project can use Vulkan GPU acceleration, not the NPU.");

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
        Console.WriteLine($"Skipping unchanged PDF: {relativePdfPath}");
        continue;
    }

    Console.WriteLine($"Indexing: {relativePdfPath}");

    // Extract plain text from every page of the PDF.
    string pdfText;
    try
    {
        pdfText = PdfContentService.ReadPdfText(file);
    }
    catch (Exception ex)
    {
        // Skip unreadable/corrupt PDFs rather than aborting the whole run.
        Console.WriteLine($"Skipping unreadable PDF: {file}");
        Console.WriteLine($"Reason: {ex.Message}");
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

Console.WriteLine($"Re-indexed {reindexedDocuments} PDFs, skipped {skippedDocuments} unchanged PDFs.");
if (reindexedDocuments > 0)
{
    Console.WriteLine($"Indexed {totalIndexedChunks} chunks.");
    if (appOptions.IncludeImages)
    {
        Console.WriteLine($"Processed {totalImageItems} embedded image items.");
    }
}

// Guard: refuse to enter the Q&A loop if there is nothing to search.
if (!VectorStoreService.HasAnyVectors(appOptions.DbPath, appOptions.CollectionName))
{
    Console.WriteLine("No text or image chunks are available in the vector database for this collection.");
    return;
}

Console.WriteLine("Ingestion complete.");
Console.WriteLine("Ask questions about the PDFs (press Enter on an empty line to exit):");

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
        Console.WriteLine("No relevant context found.");
        continue;
    }

    // Concatenate retrieved chunks into a single context block.
    // The [N] prefix and Source tag help the model cite its sources.
    var context = string.Join("\n\n---\n\n", matches.Select((m, idx) => $"[{idx + 1}] Source={m.Source}; {m.Text}"));

    // Build the RAG prompt: system instructions + retrieved context + question.
    // The model is told to rely only on the provided context, which keeps it
    // grounded and reduces hallucination on domain-specific material.
    var prompt = $"""
    You are a U.S. Air Force weather analyst assistant. Use only the following retrieved context to answer the question.
    If the context does not contain enough information, say you don't know.

    Context:
    {context}

    Question:
    {question}

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

    Console.WriteLine();
    Console.WriteLine(answer);
    Console.WriteLine();
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
