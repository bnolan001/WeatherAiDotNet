using WeatherAiDotNet.Configuration;
using WeatherAiDotNet.Models;
using WeatherAiDotNet.Services;

var appOptions = AppOptions.FromArgs(args);
var effectiveOcrCliPath = appOptions.OcrCliPath;

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

Console.WriteLine($"Using LLamaSharp with {appOptions.LlamaBackend} backend preference on this Intel Arc 140V + Intel AI Boost system.");
Console.WriteLine($"LLamaSharp tuning: gpu-layers={appOptions.GpuLayers}, threads={appOptions.Threads}, batch-threads={appOptions.BatchThreads}, batch-size={appOptions.BatchSize}, ubatch-size={appOptions.UBatchSize}.");
Console.WriteLine("Initializing LLamaSharp...");

if (!string.IsNullOrWhiteSpace(appOptions.OcrCliPath))
{
    var ocrCheck = await CliToolService.ValidateAsync(appOptions.OcrCliPath, "--version");
    if (!ocrCheck.Success)
    {
        Console.WriteLine($"OCR executable could not be validated and will be ignored: {appOptions.OcrCliPath}");
        Console.WriteLine(ocrCheck.Message);
        effectiveOcrCliPath = string.Empty;
    }
}

if (appOptions.IncludeImages)
{
    Directory.CreateDirectory(appOptions.ImagesOutputPath);
}

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

VectorStoreService.EnsureDatabase(appOptions.DbPath);

Console.WriteLine($"Found {pdfFiles.Count} PDF files.");
Console.WriteLine("Preparing embedding model...");

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

Console.WriteLine("Intel AI Boost NPU detected, but the current LLamaSharp/llama.cpp stack in this project can use Vulkan GPU acceleration, not the NPU.");

var effectiveEmbeddingSize = useModelEmbeddings ? embeddingProbe.Vector!.Length : appOptions.EmbeddingSize;
var reindexedDocuments = 0;
var skippedDocuments = 0;
var totalIndexedChunks = 0;
var totalImageItems = 0;

foreach (var file in pdfFiles)
{
    var relativePdfPath = Path.GetRelativePath(appOptions.PdfFolderPath, file);
    var fingerprint = BuildDocumentFingerprint(file);

    if (VectorStoreService.IsDocumentUpToDate(appOptions.DbPath, appOptions.CollectionName, relativePdfPath, fingerprint))
    {
        skippedDocuments++;
        Console.WriteLine($"Skipping unchanged PDF: {relativePdfPath}");
        continue;
    }

    Console.WriteLine($"Indexing: {relativePdfPath}");

    string pdfText;
    try
    {
        pdfText = PdfContentService.ReadPdfText(file);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Skipping unreadable PDF: {file}");
        Console.WriteLine($"Reason: {ex.Message}");
        continue;
    }

    var chunkIndex = 0;
    var fileVectors = new List<StoredVector>();

    var chunks = PdfContentService.ChunkText(pdfText, chunkSize: 1200, overlap: 200).ToList();
    foreach (var chunk in chunks)
    {
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

    if (appOptions.IncludeImages)
    {
        var imageItems = PdfContentService.ExtractPdfImageItems(file, appOptions.PdfFolderPath, appOptions.ImagesOutputPath, effectiveOcrCliPath);
        totalImageItems += imageItems.Count;

        foreach (var imageItem in imageItems)
        {
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
        continue;
    }

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

if (!VectorStoreService.HasAnyVectors(appOptions.DbPath, appOptions.CollectionName))
{
    Console.WriteLine("No text or image chunks are available in the vector database for this collection.");
    return;
}

Console.WriteLine("Ingestion complete.");
Console.WriteLine("Ask questions about the PDFs (press Enter on an empty line to exit):");

while (true)
{
    Console.Write("> ");
    var question = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(question))
    {
        break;
    }

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

    var context = string.Join("\n\n---\n\n", matches.Select((m, idx) => $"[{idx + 1}] Source={m.Source}; {m.Text}"));

    var prompt = $"""
    You are a U.S. Air Force weather analyst assistant. Use only the following retrieved context to answer the question.
    If the context does not contain enough information, say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """;

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

static string BuildDocumentFingerprint(string filePath)
{
    var fileInfo = new FileInfo(filePath);
    return $"{fileInfo.Length}:{fileInfo.LastWriteTimeUtc.Ticks}";
}
