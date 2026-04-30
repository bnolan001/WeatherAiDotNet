namespace WeatherAiDotNet.Tests;

/// <summary>
/// Test data for RAG verification against specific PDFs.
/// Each entry represents a question that can be answered from the PDF content,
/// and expected keywords that should appear in a correct answer.
/// </summary>
public record RagTestCase(
    string Question,
    string[] ExpectedKeywords,
    string Description);

/// <summary>
/// RAG test dataset containing known Q&A pairs for verification.
/// AFH 11-203v2: "Weather Intelligence for Operations (Unclassified)" (U.S. Air Force Handbook)
/// This document covers weather terminology, observations, forecasting, and operational applications.
/// </summary>
public static class RagTestDatasets
{
    public static readonly RagTestCase[] Afh11203v2TestCases =
    [
        new(
            "What is the definition of a synoptic observation?",
            ["synoptic", "observation", "meteorological", "regular", "intervals"],
            "Synoptic observation definition from weather terminology section"),

        new(
            "What does TAF stand for and what is its purpose?",
            ["Terminal", "Aerodrome", "Forecast", "TAF", "aeronautical", "weather"],
            "TAF definition and purpose for aviators"),

        new(
            "Explain the relationship between temperature, pressure, and altitude.",
            ["temperature", "pressure", "altitude", "atmosphere", "relationship"],
            "Atmospheric principles and physics"),

        new(
            "What are the main types of precipitation and how are they observed?",
            ["precipitation", "rain", "snow", "sleet", "hail", "observation"],
            "Precipitation types and observation methods"),

        new(
            "How is wind direction and speed reported in meteorology?",
            ["wind", "direction", "speed", "knots", "reported", "meteorological"],
            "Wind reporting standards and conventions"),

        new(
            "What is the significance of cloud types and their altitudes for flight operations?",
            ["cloud", "altitude", "flight", "operations", "visibility", "weather"],
            "Cloud classification and aviation impacts"),

        new(
            "Describe the structure and layers of the Earth's atmosphere.",
            ["atmosphere", "layers", "troposphere", "stratosphere", "altitude", "temperature"],
            "Atmospheric structure and composition"),

        new(
            "What are the primary sources of weather data used in forecasting?",
            ["weather", "data", "observations", "satellites", "radar", "forecasting"],
            "Weather data collection methods"),

        new(
            "How do meteorologists interpret pressure patterns and isobars?",
            ["pressure", "isobar", "pattern", "high", "low", "weather"],
            "Pressure analysis and weather prediction"),

        new(
            "What is the relationship between dew point and relative humidity?",
            ["dew", "point", "humidity", "relative", "moisture", "condensation"],
            "Moisture measurement relationships"),
    ];
}
