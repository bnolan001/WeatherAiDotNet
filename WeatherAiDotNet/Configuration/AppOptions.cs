namespace WeatherAiDotNet.Configuration;

internal sealed class AppOptions
{
    public required string PdfFolderPath { get; init; }
    public required string ModelPath { get; init; }
    public required string EmbeddingModelPath { get; init; }
    public required string LlamaBackend { get; init; }
    public required bool PreferGpu { get; init; }
    public required string DbPath { get; init; }
    public required string CollectionName { get; init; }
    public required int TopK { get; init; }
    public required int RetrievalPool { get; init; }
    public required int EmbeddingSize { get; init; }
    public required int GpuLayers { get; init; }
    public required int ContextSize { get; init; }
    public required bool IncludeImages { get; init; }
    public required string ImagesOutputPath { get; init; }
    public required string OcrCliPath { get; init; }

    public static AppOptions FromArgs(string[] args)
    {
        var values = CommandLineParser.Parse(args);
        var defaultDataPath = Path.Combine(AppContext.BaseDirectory, "Data");
        var defaultModelPath = Path.Combine(AppContext.BaseDirectory, "AiModels", "qwen2.5-coder-7b-instruct-q4_k_m.gguf");
        var defaultEmbeddingModelPath = Path.Combine(AppContext.BaseDirectory, "AiModels", "bge-base-en-v1.5-q4_k_m.gguf");

        return new AppOptions
        {
            PdfFolderPath = CommandLineParser.GetOption(values, "pdf-folder", defaultDataPath),
            ModelPath = CommandLineParser.GetOption(values, "model-path", defaultModelPath),
            EmbeddingModelPath = CommandLineParser.GetOption(values, "embedding-model-path", defaultEmbeddingModelPath),
            LlamaBackend = CommandLineParser.GetOption(values, "llama-backend", "vulkan"),
            PreferGpu = !bool.TryParse(CommandLineParser.GetOption(values, "prefer-gpu", "true"), out var preferGpu) || preferGpu,
            DbPath = CommandLineParser.GetOption(values, "db-path", "rag-vectors.db"),
            CollectionName = CommandLineParser.GetOption(values, "collection", "pdf-rag-poc"),
            TopK = int.TryParse(CommandLineParser.GetOption(values, "top-k", "6"), out var topK) ? topK : 6,
            RetrievalPool = int.TryParse(CommandLineParser.GetOption(values, "retrieval-pool", "24"), out var retrievalPool) ? retrievalPool : 24,
            EmbeddingSize = int.TryParse(CommandLineParser.GetOption(values, "embedding-size", "384"), out var embeddingSize) ? embeddingSize : 384,
            GpuLayers = int.TryParse(CommandLineParser.GetOption(values, "gpu-layers", "20"), out var gpuLayers) ? gpuLayers : 20,
            ContextSize = int.TryParse(CommandLineParser.GetOption(values, "ctx-size", "4096"), out var contextSize) ? contextSize : 4096,
            IncludeImages = bool.TryParse(CommandLineParser.GetOption(values, "include-images", "true"), out var includeImages) && includeImages,
            ImagesOutputPath = CommandLineParser.GetOption(values, "images-output", Path.Combine(AppContext.BaseDirectory, "extracted-images")),
            OcrCliPath = CommandLineParser.GetOption(values, "ocr-cli", string.Empty)
        };
    }
}