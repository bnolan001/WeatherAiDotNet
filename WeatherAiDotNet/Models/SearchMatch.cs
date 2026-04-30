namespace WeatherAiDotNet.Models;

/// <summary>
/// Represents a single retrieved chunk from the vector database along with its
/// combined relevance score. Returned by <see cref="Services.VectorStoreService.Search"/>
/// and used to build the context window that is injected into the LLM prompt.
/// </summary>
/// <param name="Source">Relative file path of the PDF that this chunk came from.</param>
/// <param name="ChunkIndex">Zero-based position of this chunk within its source document.</param>
/// <param name="Text">The raw text content of the chunk.</param>
/// <param name="Score">
/// Combined hybrid relevance score (80 % cosine similarity + 20 % lexical overlap).
/// Higher scores indicate a closer match to the user's question.
/// </param>
public record SearchMatch(string Source, int ChunkIndex, string Text, float Score);