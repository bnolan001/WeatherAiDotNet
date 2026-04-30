namespace WeatherAiDotNet.Configuration;

/// <summary>
/// Parses <c>--key value</c> style command-line arguments into a dictionary.
/// Arguments that appear without a following value (e.g., a lone flag) are stored
/// with an implicit value of <c>"true"</c> so they can be treated as boolean switches.
/// </summary>
internal static class CommandLineParser
{
    /// <summary>
    /// Iterates <paramref name="args"/> and builds a case-insensitive key/value map
    /// from consecutive <c>--key value</c> pairs.
    /// </summary>
    /// <param name="args">The raw command-line argument array from <c>string[] args</c>.</param>
    /// <returns>
    /// A dictionary where each key is the argument name without the leading <c>--</c>
    /// and each value is the string that followed it, or <c>"true"</c> for bare flags.
    /// </returns>
    public static Dictionary<string, string> Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < args.Length; i++)
        {
            var current = args[i];

            // Only process tokens that start with "--"; skip positional arguments.
            if (!current.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            var key = current[2..]; // strip the leading "--"

            // If the next token exists and is not itself a flag, consume it as the value.
            var value = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal)
                ? args[++i]
                : "true";

            values[key] = value;
        }

        return values;
    }

    /// <summary>
    /// Looks up a parsed option by key, returning <paramref name="fallback"/> when
    /// the key is absent. Comparison is case-insensitive.
    /// </summary>
    /// <param name="options">The dictionary returned by <see cref="Parse"/>.</param>
    /// <param name="key">The option name to look up (without the <c>--</c> prefix).</param>
    /// <param name="fallback">The default value to return when the key is not found.</param>
    /// <returns>The parsed value if present, otherwise <paramref name="fallback"/>.</returns>
    public static string GetOption(IReadOnlyDictionary<string, string> options, string key, string fallback)
        => options.TryGetValue(key, out var value) ? value : fallback;
}