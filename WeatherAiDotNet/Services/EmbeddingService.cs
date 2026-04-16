using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

internal static class EmbeddingService
{
    public static async Task<float[]> GetEmbeddingAsync(
        string input,
        int fallbackEmbeddingSize,
        bool useModelEmbeddings,
        string llamaEmbeddingCliPath,
        string embeddingModelPath,
        int gpuLayers,
        int contextSize)
    {
        if (useModelEmbeddings)
        {
            var vector = await GenerateEmbeddingWithLlamaCppAsync(
                llamaEmbeddingCliPath,
                embeddingModelPath,
                input,
                gpuLayers,
                contextSize);

            if (vector is { Length: > 0 })
            {
                Normalize(vector);
                return vector;
            }
        }

        return CreateLocalEmbedding(input, fallbackEmbeddingSize);
    }

    public static async Task<EmbeddingProbeResult> ProbeAsync(
        string llamaEmbeddingCliPath,
        string embeddingModelPath,
        int gpuLayers,
        int contextSize)
        => await TryGenerateEmbeddingWithDiagnosticsAsync(
            llamaEmbeddingCliPath,
            embeddingModelPath,
            "embedding calibration",
            gpuLayers,
            contextSize);

    private static async Task<float[]?> GenerateEmbeddingWithLlamaCppAsync(
        string llamaEmbeddingCliPath,
        string embeddingModelPath,
        string input,
        int gpuLayers,
        int contextSize)
    {
        var result = await TryGenerateEmbeddingWithDiagnosticsAsync(
            llamaEmbeddingCliPath,
            embeddingModelPath,
            input,
            gpuLayers,
            contextSize);

        return result.Vector;
    }

    private static async Task<EmbeddingProbeResult> TryGenerateEmbeddingWithDiagnosticsAsync(
        string llamaEmbeddingCliPath,
        string embeddingModelPath,
        string input,
        int gpuLayers,
        int contextSize)
    {
        var promptFilePath = Path.Combine(Path.GetTempPath(), $"rag-embed-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(promptFilePath, input, Encoding.UTF8);

        var profiles = new List<string[]>
        {
            new[] { "--embd-normalize", "2", "--embd-output-format", "json", "--log-disable" },
            new[] { "--embd-output-format", "json", "--log-disable" },
            new[] { "--log-disable" },
            Array.Empty<string>()
        };

        string? lastDiagnostic = null;

        try
        {
            foreach (var profile in profiles)
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = llamaEmbeddingCliPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                startInfo.ArgumentList.Add("-m");
                startInfo.ArgumentList.Add(embeddingModelPath);
                startInfo.ArgumentList.Add("-ngl");
                startInfo.ArgumentList.Add(Math.Max(0, gpuLayers).ToString());
                startInfo.ArgumentList.Add("-c");
                startInfo.ArgumentList.Add(Math.Max(512, contextSize).ToString());
                startInfo.ArgumentList.Add("-f");
                startInfo.ArgumentList.Add(promptFilePath);

                foreach (var arg in profile)
                {
                    startInfo.ArgumentList.Add(arg);
                }

                using var process = new Process { StartInfo = startInfo };

                try
                {
                    process.Start();
                }
                catch (Exception ex)
                {
                    return new EmbeddingProbeResult(null, ex.Message);
                }

                var outputTask = process.StandardOutput.ReadToEndAsync();
                var errorTask = process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();

                var output = await outputTask;
                var error = await errorTask;

                if (process.ExitCode != 0)
                {
                    lastDiagnostic = string.IsNullOrWhiteSpace(error)
                        ? $"Embedding process exited with code {process.ExitCode}."
                        : error.Trim();
                    continue;
                }

                if (string.IsNullOrWhiteSpace(output))
                {
                    lastDiagnostic = "Embedding process returned no output.";
                    continue;
                }

                var vector = TryParseEmbeddingVector(output);
                if (vector is { Length: > 0 })
                {
                    return new EmbeddingProbeResult(vector, null);
                }

                lastDiagnostic = "Embedding output could not be parsed.";
            }
        }
        finally
        {
            TryDelete(promptFilePath);
        }

        return new EmbeddingProbeResult(null, lastDiagnostic ?? "Unknown embedding error.");
    }

    private static float[]? TryParseEmbeddingVector(string output)
    {
        try
        {
            using var document = JsonDocument.Parse(output);
            var parsed = TryParseEmbeddingVectorFromElement(document.RootElement);
            if (parsed is { Length: > 0 })
            {
                return parsed;
            }
        }
        catch
        {
        }

        var start = output.IndexOf('[');
        var end = output.LastIndexOf(']');

        if (start >= 0 && end > start)
        {
            var jsonSlice = output[start..(end + 1)];
            try
            {
                using var document = JsonDocument.Parse(jsonSlice);
                var parsed = TryParseEmbeddingVectorFromElement(document.RootElement);
                if (parsed is { Length: > 0 })
                {
                    return parsed;
                }
            }
            catch
            {
            }
        }

        return TryParseEmbeddingVectorFromPlainText(output);
    }

    private static float[]? TryParseEmbeddingVectorFromPlainText(string output)
    {
        var separators = new[] { ' ', '\t', '\r', '\n', ',', ';', '[', ']' };
        var values = new List<float>();

        foreach (var token in output.Split(separators, StringSplitOptions.RemoveEmptyEntries))
        {
            if (float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
            {
                values.Add(value);
            }
        }

        return values.Count >= 32 ? values.ToArray() : null;
    }

    private static float[]? TryParseEmbeddingVectorFromElement(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Array)
        {
            if (element.GetArrayLength() == 0)
            {
                return null;
            }

            var first = element[0];
            if (first.ValueKind == JsonValueKind.Number)
            {
                return element.EnumerateArray().Select(x => x.GetSingle()).ToArray();
            }

            if (first.ValueKind == JsonValueKind.Array)
            {
                return first.EnumerateArray().Select(x => x.GetSingle()).ToArray();
            }
        }

        if (element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty("embedding", out var embedding)
                && embedding.ValueKind == JsonValueKind.Array)
            {
                return embedding.EnumerateArray().Select(x => x.GetSingle()).ToArray();
            }

            if (element.TryGetProperty("data", out var data)
                && data.ValueKind == JsonValueKind.Array
                && data.GetArrayLength() > 0)
            {
                var firstItem = data[0];
                if (firstItem.ValueKind == JsonValueKind.Object
                    && firstItem.TryGetProperty("embedding", out var itemEmbedding)
                    && itemEmbedding.ValueKind == JsonValueKind.Array)
                {
                    return itemEmbedding.EnumerateArray().Select(x => x.GetSingle()).ToArray();
                }
            }
        }

        return null;
    }

    private static float[] CreateLocalEmbedding(string input, int size)
    {
        var vector = new float[size];

        foreach (var token in Tokenize(input))
        {
            var hash = StableTokenHash(token);
            var index = (int)(hash % (uint)size);
            var sign = (hash & 1U) == 0U ? 1f : -1f;
            vector[index] += sign;
        }

        Normalize(vector);
        return vector;
    }

    private static IEnumerable<string> Tokenize(string input)
    {
        var sb = new StringBuilder();

        foreach (var ch in input)
        {
            if (char.IsLetterOrDigit(ch))
            {
                sb.Append(char.ToLowerInvariant(ch));
                continue;
            }

            if (sb.Length > 0)
            {
                var token = sb.ToString();
                if (token.Length >= 2)
                {
                    yield return token;
                }

                sb.Clear();
            }
        }

        if (sb.Length > 0)
        {
            var token = sb.ToString();
            if (token.Length >= 2)
            {
                yield return token;
            }
        }
    }

    private static uint StableTokenHash(string token)
    {
        const uint offsetBasis = 2166136261;
        const uint prime = 16777619;

        var hash = offsetBasis;
        foreach (var ch in token)
        {
            hash ^= ch;
            hash *= prime;
        }

        return hash;
    }

    private static void Normalize(float[] vector)
    {
        double sum = 0;
        for (var i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        if (sum <= 0)
        {
            return;
        }

        var norm = (float)Math.Sqrt(sum);
        for (var i = 0; i < vector.Length; i++)
        {
            vector[i] /= norm;
        }
    }

    private static void TryDelete(string filePath)
    {
        try
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
        catch
        {
        }
    }
}