using LLama.Common;
using LLama.Native;

namespace WeatherAiDotNet.Services;

internal static class LlamaNativeService
{
    private static readonly object SyncRoot = new();
    private static bool s_configured;

    public static void Configure(string backend, bool preferGpu)
    {
        lock (SyncRoot)
        {
            if (s_configured)
            {
                return;
            }

            var useVulkan = preferGpu && string.Equals(backend, "vulkan", StringComparison.OrdinalIgnoreCase);
            NativeLibraryConfig.All
                .WithCuda(false)
                .WithVulkan(useVulkan)
                .WithAutoFallback(true);

            s_configured = true;
        }
    }

    public static string GetRequestedRuntimeBackend(string backend, bool preferGpu, int gpuLayers)
        => preferGpu
            && gpuLayers > 0
            && string.Equals(backend, "vulkan", StringComparison.OrdinalIgnoreCase)
                ? "vulkan"
                : "cpu";

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

    public static ModelParams CreateEmbeddingModelParams(string modelPath, int gpuLayers, int contextSize, int threads, int batchThreads, int batchSize, int uBatchSize)
        => new(modelPath)
        {
            ContextSize = (uint)Math.Max(512, contextSize),
            GpuLayerCount = Math.Max(0, gpuLayers),
            Threads = Math.Max(1, threads),
            BatchThreads = Math.Max(1, batchThreads),
            BatchSize = (uint)Math.Max(32, batchSize),
            UBatchSize = (uint)Math.Max(32, uBatchSize),
            Embeddings = true,
            PoolingType = LLamaPoolingType.Mean
        };
}
