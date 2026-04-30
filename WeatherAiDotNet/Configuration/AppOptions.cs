namespace WeatherAiDotNet.Configuration;

/// <summary>
/// Holds all runtime configuration for the application.
/// Values come from command-line arguments and fall back to sensible defaults when
/// an argument is not supplied.  Use <see cref="FromArgs"/> to construct an instance.
/// </summary>
/// <remarks>
/// Every property is <c>required</c> and <c>init</c>-only, making the object immutable
/// after construction.  This prevents accidental mutation during the ingestion or
/// Q&amp;A phases of the pipeline.
/// </remarks>
public class AppOptions
{
    /// <summary>
    /// Path to the folder containing PDF files to index.
    /// Default: {AppDirectory}/Data
    /// Command-line: --pdf-folder
    /// </summary>
    public required string PdfFolderPath { get; init; }

    /// <summary>
    /// Path to the main LLM model file (GGUF format).
    /// Used for generating answers to user questions.
    /// Default: {AppDirectory}/AiModels/qwen2.5-coder-7b-instruct-q4_k_m.gguf
    /// Command-line: --model-path
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// Path to the embedding model file (GGUF format).
    /// Used for generating vector embeddings of text chunks for semantic search.
    /// Default: {AppDirectory}/AiModels/bge-base-en-v1.5-q4_k_m.gguf
    /// Command-line: --embedding-model-path
    /// </summary>
    public required string EmbeddingModelPath { get; init; }

    /// <summary>
    /// LLamaSharp backend preference: "vulkan" or "cpu".
    /// Determines whether to attempt GPU acceleration via Vulkan (Intel Arc/Iris).
    /// Falls back to CPU if the preferred backend is unavailable.
    /// Default: "vulkan"
    /// Command-line: --llama-backend
    /// </summary>
    public required string LlamaBackend { get; init; }

    /// <summary>
    /// Whether to prefer GPU acceleration when available.
    /// If false, forces CPU-only execution regardless of backend setting.
    /// Default: true
    /// Command-line: --prefer-gpu
    /// </summary>
    public required bool PreferGpu { get; init; }

    /// <summary>
    /// Database path for storing vector embeddings.
    /// SQLite database used for semantic search and document tracking.
    /// Default: "rag-vectors.db"
    /// Command-line: --db-path
    /// </summary>
    public required string DbPath { get; init; }

    /// <summary>
    /// Collection name within the vector database.
    /// Allows organizing multiple RAG datasets within a single database.
    /// Default: "pdf-rag-poc"
    /// Command-line: --collection
    /// </summary>
    public required string CollectionName { get; init; }

    /// <summary>
    /// Number of top K similar chunks to retrieve for context during answer generation.
    /// Higher values provide more context but may dilute relevance.
    /// Typical range: 3-10
    /// Default: 6
    /// Command-line: --top-k
    /// </summary>
    public required int TopK { get; init; }

    /// <summary>
    /// Number of candidate chunks to search before selecting top-k results.
    /// Larger pool enables better ranking but slower search.
    /// Should be larger than TopK.
    /// Default: 24
    /// Command-line: --retrieval-pool
    /// </summary>
    public required int RetrievalPool { get; init; }

    /// <summary>
    /// Dimension size for fallback hash-based embeddings when model embeddings unavailable.
    /// Must match the actual embedding model's output dimension.
    /// Default: 384 (matches bge-base-en-v1.5)
    /// Command-line: --embedding-size
    /// </summary>
    public required int EmbeddingSize { get; init; }

    /// <summary>
    /// Number of model layers to offload to GPU.
    /// Higher values use more GPU memory but improve speed.
    /// Set to 999 to offload all layers that fit in VRAM.
    /// Default: 999
    /// Command-line: --gpu-layers
    /// </summary>
    public required int GpuLayers { get; init; }

    /// <summary>
    /// Token context window size (maximum tokens the model can see at once).
    /// Larger contexts improve understanding but use more memory.
    /// Must be compatible with the model's training context length.
    /// Default: 4096
    /// Command-line: --ctx-size
    /// </summary>
    public required int ContextSize { get; init; }

    /// <summary>
    /// Number of CPU threads for model inference.
    /// Defaults to logical processor count.
    /// Lower values reduce CPU load; higher values may improve throughput on high-core systems.
    /// Default: Environment.ProcessorCount
    /// Command-line: --threads
    /// </summary>
    public required int Threads { get; init; }

    /// <summary>
    /// Number of CPU threads for batch processing.
    /// Improves throughput when processing multiple tokens/sequences in parallel.
    /// Defaults to logical processor count.
    /// Default: Environment.ProcessorCount
    /// Command-line: --batch-threads
    /// </summary>
    public required int BatchThreads { get; init; }

    /// <summary>
    /// Batch size for token processing (max tokens per batch).
    /// Larger batches improve GPU utilization but use more memory.
    /// Typical range: 256-1024 depending on model and VRAM.
    /// Default: 512
    /// Command-line: --batch-size
    /// </summary>
    public required int BatchSize { get; init; }

    /// <summary>
    /// Micro-batch size for unified batch processing.
    /// Balances between GPU efficiency and memory usage.
    /// Usually smaller than BatchSize.
    /// Typical range: 32-256.
    /// Default: 256
    /// Command-line: --ubatch-size
    /// </summary>
    public required int UBatchSize { get; init; }

    /// <summary>
    /// Whether to extract and index images from PDFs.
    /// Requires OCR tool for image text extraction (if --ocr-cli is set).
    /// Default: true
    /// Command-line: --include-images
    /// </summary>
    public required bool IncludeImages { get; init; }

    /// <summary>
    /// Output directory for extracted images from PDFs.
    /// Only used if IncludeImages is true.
    /// Default: {AppDirectory}/extracted-images
    /// Command-line: --images-output
    /// </summary>
    public required string ImagesOutputPath { get; init; }

    /// <summary>
    /// Path to OCR command-line tool (e.g., tesseract.exe).
    /// Optional; if not provided or invalid, image OCR is skipped.
    /// Default: "" (empty, OCR disabled)
    /// Command-line: --ocr-cli
    /// </summary>
    public required string OcrCliPath { get; init; }

    /// <summary>
    /// Creates an instance of <see cref="AppOptions"/> by parsing the specified command-line arguments.
    /// </summary>
    /// <param name="args">The command-line arguments to parse.</param>
    /// <returns>A new instance of <see cref="AppOptions"/> with properties set according to the parsed arguments.</returns>
    public static AppOptions FromArgs(string[] args)
    {
        // Parse all --key value pairs from the command line into a lookup dictionary.
        var values = CommandLineParser.Parse(args);

        // Build default paths relative to the application's own directory so the
        // project works out of the box without any arguments.
        var defaultDataPath = Path.Combine(AppContext.BaseDirectory, "Data");
        var defaultModelPath = Path.Combine(AppContext.BaseDirectory, "AiModels", "qwen2.5-coder-7b-instruct-q4_k_m.gguf");
        var defaultEmbeddingModelPath = Path.Combine(AppContext.BaseDirectory, "AiModels", "bge-base-en-v1.5-q4_k_m.gguf");

        // Use all logical processors by default; clamped to at least 1 below.
        var defaultThreads = Math.Max(1, Environment.ProcessorCount);

        return new AppOptions
        {
            PdfFolderPath = CommandLineParser.GetOption(values, "pdf-folder", defaultDataPath),
            ModelPath = CommandLineParser.GetOption(values, "model-path", defaultModelPath),
            EmbeddingModelPath = CommandLineParser.GetOption(values, "embedding-model-path", defaultEmbeddingModelPath),
            LlamaBackend = CommandLineParser.GetOption(values, "llama-backend", "vulkan"),

            // --prefer-gpu defaults to true; any parse failure also keeps the default.
            PreferGpu = !bool.TryParse(CommandLineParser.GetOption(values, "prefer-gpu", "true"), out var preferGpu) || preferGpu,
            DbPath = CommandLineParser.GetOption(values, "db-path", "rag-vectors.db"),
            CollectionName = CommandLineParser.GetOption(values, "collection", "pdf-rag-poc"),
            TopK = int.TryParse(CommandLineParser.GetOption(values, "top-k", "6"), out var topK) ? topK : 6,
            RetrievalPool = int.TryParse(CommandLineParser.GetOption(values, "retrieval-pool", "24"), out var retrievalPool) ? retrievalPool : 24,
            EmbeddingSize = int.TryParse(CommandLineParser.GetOption(values, "embedding-size", "384"), out var embeddingSize) ? embeddingSize : 384,
            GpuLayers = int.TryParse(CommandLineParser.GetOption(values, "gpu-layers", "999"), out var gpuLayers) ? gpuLayers : 999,
            ContextSize = int.TryParse(CommandLineParser.GetOption(values, "ctx-size", "4096"), out var contextSize) ? contextSize : 4096,

            // Clamp thread counts to at least 1 to prevent invalid LLamaSharp configurations.
            Threads = int.TryParse(CommandLineParser.GetOption(values, "threads", defaultThreads.ToString()), out var threads) ? Math.Max(1, threads) : defaultThreads,
            BatchThreads = int.TryParse(CommandLineParser.GetOption(values, "batch-threads", defaultThreads.ToString()), out var batchThreads) ? Math.Max(1, batchThreads) : defaultThreads,

            // Clamp batch sizes to sane minimums to avoid LLamaSharp assertion failures.
            BatchSize = int.TryParse(CommandLineParser.GetOption(values, "batch-size", "512"), out var batchSize) ? Math.Max(32, batchSize) : 512,
            UBatchSize = int.TryParse(CommandLineParser.GetOption(values, "ubatch-size", "256"), out var ubatchSize) ? Math.Max(32, ubatchSize) : 256,
            IncludeImages = bool.TryParse(CommandLineParser.GetOption(values, "include-images", "true"), out var includeImages) && includeImages,
            ImagesOutputPath = CommandLineParser.GetOption(values, "images-output", Path.Combine(AppContext.BaseDirectory, "extracted-images")),
            OcrCliPath = CommandLineParser.GetOption(values, "ocr-cli", string.Empty)
        };
    }
}