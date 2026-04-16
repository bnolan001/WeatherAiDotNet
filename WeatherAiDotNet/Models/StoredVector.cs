namespace WeatherAiDotNet.Models;

internal sealed record StoredVector(string Collection, string Source, int ChunkIndex, string Text, float[] Vector);