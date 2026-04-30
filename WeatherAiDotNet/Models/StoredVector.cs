namespace WeatherAiDotNet.Models;

/// <summary>
/// Represents a single indexed chunk of text and its associated embedding vector.
/// Each PDF is split into many overlapping chunks, and each chunk is stored as a
/// <see cref="StoredVector"/> in the SQLite database so it can be retrieved later
/// via semantic similarity search.
/// </summary>
/// <param name="Collection">Logical grouping name for the dataset (e.g., "pdf-rag-poc").</param>
/// <param name="Source">Relative file path of the PDF that produced this chunk.</param>
/// <param name="ChunkIndex">Zero-based position of this chunk within its source document.</param>
/// <param name="Text">The raw text content of the chunk that was embedded.</param>
/// <param name="Vector">The floating-point embedding vector produced from <see cref="Text"/>.</param>
public record StoredVector(string Collection, string Source, int ChunkIndex, string Text, float[] Vector);