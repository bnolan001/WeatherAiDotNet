using System.Text;
using LLama;
using LLama.Common;
using LLama.Sampling;

namespace WeatherAiDotNet.Services;

internal static class LlamaGenerationService
{
    private static readonly object SyncRoot = new();
    private static LLamaWeights? s_weights;
    private static StatelessExecutor? s_executor;
    private static string? s_modelPath;
    private static string? s_backend;
    private static bool s_preferGpu;
    private static int s_gpuLayers;
    private static int s_contextSize;

    public static async Task<string> GenerateAnswerAsync(
        string modelPath,
        string prompt,
        int maxTokens,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize)
    {
        EnsureInitialized(modelPath, backend, preferGpu, gpuLayers, contextSize);

        var inferenceParams = new InferenceParams
        {
            MaxTokens = maxTokens,
            AntiPrompts = ["\nQuestion:", "\nUser:", "<|im_end|>"],
            SamplingPipeline = new DefaultSamplingPipeline()
        };

        var response = new StringBuilder();
        await foreach (var text in s_executor!.InferAsync(prompt, inferenceParams))
        {
            response.Append(text);
        }

        return response.Length == 0
            ? "No response returned by local model."
            : response.ToString().Trim();
    }

    private static void EnsureInitialized(string modelPath, string backend, bool preferGpu, int gpuLayers, int contextSize)
    {
        lock (SyncRoot)
        {
            if (s_executor is not null
                && string.Equals(s_modelPath, modelPath, StringComparison.OrdinalIgnoreCase)
                && string.Equals(s_backend, backend, StringComparison.OrdinalIgnoreCase)
                && s_preferGpu == preferGpu
                && s_gpuLayers == gpuLayers
                && s_contextSize == contextSize)
            {
                return;
            }

            DisposeCurrent();
            LlamaNativeService.Configure(backend, preferGpu);

            try
            {
                var modelParams = LlamaNativeService.CreateGenerationModelParams(modelPath, gpuLayers, contextSize);
                s_weights = LLamaWeights.LoadFromFile(modelParams);
                s_executor = new StatelessExecutor(s_weights, modelParams, logger: null);
            }
            catch when (preferGpu && gpuLayers > 0)
            {
                var cpuParams = LlamaNativeService.CreateGenerationModelParams(modelPath, 0, contextSize);
                s_weights = LLamaWeights.LoadFromFile(cpuParams);
                s_executor = new StatelessExecutor(s_weights, cpuParams, logger: null);
            }

            s_modelPath = modelPath;
            s_backend = backend;
            s_preferGpu = preferGpu;
            s_gpuLayers = gpuLayers;
            s_contextSize = contextSize;
        }
    }

    private static void DisposeCurrent()
    {
        s_executor?.Context.Dispose();
        s_weights?.Dispose();
        s_executor = null;
        s_weights = null;
        s_modelPath = null;
        s_backend = null;
        s_preferGpu = false;
        s_gpuLayers = 0;
        s_contextSize = 0;
    }
}