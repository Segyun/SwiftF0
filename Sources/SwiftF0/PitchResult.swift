// Copyright Huigyun Jeong. All rights reserved.

/// Container of F0 detection result
public struct PitchResult {
  /// Detected pitch frequency in Hertz (Hz)
  public let pitchHz: Float
  /// Confidence level of the pitch detection (0.0 to 1.0)
  public let confidence: Float
  /// Timestamp of the pitch detection in seconds
  public let timeStamp: Float
  /// Indicates whether the detected pitch is voiced (true) or unvoiced (false)
  public let voicing: Bool
}
