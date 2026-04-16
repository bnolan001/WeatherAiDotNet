namespace WeatherAiDotNet.Models;

internal sealed record SearchMatch(string Source, int ChunkIndex, string Text, float Score);