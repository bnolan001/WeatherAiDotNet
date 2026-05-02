using LLama.Common;
using LLama.Native;
using Microsoft.Extensions.Logging;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Centralises LLamaSharp native-library configuration and <see cref="ModelParams"/>
/// creation so that both <see cref="EmbeddingService"/> and <see cref="LlamaGenerationService"/>
/// use identical settings and the native backend is only configured once.
/// </summary>
/// <remarks>
/// LLamaSharp must be told which native backend (Vulkan, CUDA, CPU …) to load
/// before any model is opened; calling <see cref="Configure"/> more than once would
/// be a no-op due to the internal <c>s_configured</c> guard.
/// <para>
/// Call <see cref="ConfigureLogging"/> once at startup (before any call to
/// <see cref="Configure"/>) to route all llama.cpp native log messages through the
/// application's Serilog-backed <see cref="ILogger"/> pipeline.
/// </para>
/// </remarks>
public static class LlamaNativeService
{
    private static readonly object SyncRoot = new();

    // Ensures Configure() only applies NativeLibraryConfig settings once per process.
    private static bool s_configured;

    // Holds the ILogger supplied by Program.cs for use inside Configure().
    // Must be set before the first call to Configure().
    private static ILogger? s_logger;

    /// <summary>
    /// Stores the <see cref="ILogger"/> that will be forwarded to
    /// <c>NativeLibraryConfig.All.WithLogCallback</c> so that every llama.cpp
    /// native log message flows through the application's Serilog pipeline.
    /// </summary>
    /// <remarks>
    /// This must be called before <see cref="Configure"/> because the log callback
    /// is registered as part of the one-time native library setup and cannot be
    /// changed after the library has loaded.
    /// </remarks>
    /// <param name="loggerFactory">The shared logger factory, typically Serilog-backed.</param>
    public static void ConfigureLogging(ILoggerFactory loggerFactory)
    {
        lock (SyncRoot)
        {
            s_logger = loggerFactory.CreateLogger(typeof(LlamaNativeService).FullName!);
        }
    }

    /// <summary>
    /// Applies the LLamaSharp native-library backend preferences for the current
    /// process.  Must be called before loading any model weights.
    /// </summary>
    /// <param name="backend">
    /// Requested backend name, e.g. <c>"vulkan"</c> or <c>"cpu"</c>.
    /// Only <c>"vulkan"</c> triggers GPU acceleration; anything else defaults to CPU.
    /// </param>
    /// <param name="preferGpu">
    /// When <see langword="false"/> the Vulkan path is skipped even if
    /// <paramref name="backend"/> is <c>"vulkan"</c>.
    /// </param>
    public static void Configure(string backend, bool preferGpu)
    {
        lock (SyncRoot)
        {
            if (s_configured)
            {
                return;
            }

            var useVulkan = preferGpu && string.Equals(backend, "vulkan", StringComparison.OrdinalIgnoreCase);

            var config = NativeLibraryConfig.All
                .WithCuda(false)
                .WithVulkan(useVulkan)
                .WithAutoFallback(true);

            // Route all llama.cpp native log messages through the application logger
            // so they appear in the same Serilog rolling file as the rest of the app.
            // The ILogger overload maps llama.cpp log levels to the corresponding
            // Microsoft.Extensions.Logging levels automatically.
            if (s_logger is not null)
            {
                config.WithLogCallback(s_logger);
            }

            s_configured = true;
        }
    }

    /// <summary>
    /// Returns a human-readable label describing which backend will actually be used
    /// based on the supplied settings.
    /// </summary>
    /// <param name="backend">Requested backend name.</param>
    /// <param name="preferGpu">Whether GPU usage is requested.</param>
    /// <param name="gpuLayers">Number of layers to offload to the GPU.</param>
    /// <returns><c>"vulkan"</c> when GPU acceleration is expected; otherwise <c>"cpu"</c>.</returns>
    public static string GetRequestedRuntimeBackend(string backend, bool preferGpu, int gpuLayers)
        => preferGpu
            && gpuLayers > 0
            && string.Equals(backend, "vulkan", StringComparison.OrdinalIgnoreCase)
                ? "vulkan"
                : "cpu";

    /// <summary>
    /// Creates <see cref="ModelParams"/> tuned for text-generation inference.
    /// All integer values are clamped to their minimum safe levels.
    /// </summary>
    public static ModelParams CreateGenerationModelParams(string modelPath, int gpuLayers, int contextSize, int threads, int batchThreads, int batchSize, int uBatchSize)
        => new(modelPath)
        {
            ContextSize = (uint)Math.Max(512, contextSize),
            GpuLayerCount = Math.Max(0, gpuLayers),
            Threads = Math.Max(1, threads),
            BatchThreads = Math.Max(1, batchThreads),
            BatchSize = (uint)Math.Max(32, batchSize),
            UBatchSize = (uint)Math.Max(32, uBatchSize)
        };

    /// <summary>
    /// Creates <see cref="ModelParams"/> tuned for embedding inference.
    /// Sets <c>Embeddings = true</c> and <c>PoolingType = Mean</c> so the model
    /// returns a single fixed-length vector per input rather than per-token logits.
    /// Mean pooling averages all token vectors, producing a stable sentence-level
    /// representation that works well for cosine-similarity search.
    /// </summary>
    public static ModelParams CreateEmbeddingModelParams(string modelPath, int gpuLayers, int contextSize, int threads, int batchThreads, int batchSize, int uBatchSize)
        => new(modelPath)
        {
            ContextSize = (uint)Math.Max(512, contextSize),
            GpuLayerCount = Math.Max(0, gpuLayers),
            Threads = Math.Max(1, threads),
            BatchThreads = Math.Max(1, batchThreads),
            BatchSize = (uint)Math.Max(32, batchSize),
            UBatchSize = (uint)Math.Max(32, uBatchSize),
            Embeddings = true,              // must be true for the embedding model to return vectors
            PoolingType = LLamaPoolingType.Mean  // average all token embeddings into one sentence vector
        };
}
