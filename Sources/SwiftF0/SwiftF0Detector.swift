// Copyright Huigyun Jeong. All rights reserved.

@preconcurrency import AVFAudio
import Foundation
import OnnxRuntimeBindings

/// Error for `SwiftF0Detector`
public enum SwiftF0DetectorError: Error, LocalizedError {
  case invalidConfidenceThreshold
  case invalidFrequency
  case incompatibleModelFormat
  case failedToConvertAudioFile
  case failedToDetectPitch

  /// Provides localized descriptions for each error case
  public var errorDescription: String? {
    switch self {
    case .invalidConfidenceThreshold:
      return "The confidence threshold must be between 0.0 and 1.0."
    case .invalidFrequency:
      return
        "The specified frequency range is invalid or out of supported bounds."
    case .incompatibleModelFormat:
      return "The provided model format is incompatible with the detector."
    case .failedToConvertAudioFile:
      return "Failed to convert the audio file to the required format."
    case .failedToDetectPitch:
      return "Pitch detection failed due to unexpected output from the model."
    }
  }
}

/// F0 detector using SwiftF0
public final class SwiftF0Detector {
  private enum Constants {
    static let inputName: String = "input_audio"
    static let outputNames: [String] = ["pitch_hz", "confidence"]

    static let targetSampleRate: Double = 16_000
    static let hopSize: Int = 256
    static let frameSize: Int = 1024
    static let stftPadding: Int = (frameSize - hopSize) / 2
    static let minAudioLength: Int = 256

    static let centerOffset: Double =
      Double(frameSize - 1) / 2 - Double(stftPadding)

    static let minFrequency: Float = 46.875
    static let maxFrequency: Float = 2093.75
  }

  private let confidenceThreshold: Float
  private let minFrequency: Float
  private let maxFrequency: Float

  private let ortEnv: ORTEnv
  private let ortSession: ORTSession

  /// Initializes the SwiftF0Detector with a model URL and optional parameters.
  /// - Throws: Errors if parameters are invalid or model loading fails.
  public init(
    modelUrl: URL,
    confidenceThreshold: Float = 0.9,
    minFrequency: Float? = nil,
    maxFrequency: Float? = nil
  ) throws {
    self.confidenceThreshold = confidenceThreshold

    // Validate confidence threshold is within [0, 1]
    guard 0.0 <= self.confidenceThreshold, self.confidenceThreshold <= 1.0
    else {
      throw SwiftF0DetectorError.invalidConfidenceThreshold
    }

    self.minFrequency = minFrequency ?? Constants.minFrequency
    self.maxFrequency = maxFrequency ?? Constants.maxFrequency

    // Validate frequency bounds
    guard self.minFrequency >= Constants.minFrequency,
      self.minFrequency < Constants.maxFrequency
    else {
      throw SwiftF0DetectorError.invalidFrequency
    }

    guard self.maxFrequency <= Constants.maxFrequency,
      self.maxFrequency > Constants.minFrequency
    else {
      throw SwiftF0DetectorError.invalidFrequency
    }

    guard self.minFrequency < self.maxFrequency else {
      throw SwiftF0DetectorError.invalidFrequency
    }

    // Initialize ONNX Runtime environment
    let ortEnv: ORTEnv = try ORTEnv(loggingLevel: .warning)

    self.ortEnv = ortEnv

    // Setup session options with CoreML execution provider
    let ortSessionOptions: ORTSessionOptions = try ORTSessionOptions()
    let ortCoreMLExecutionProviderOptions: ORTCoreMLExecutionProviderOptions =
      ORTCoreMLExecutionProviderOptions()
    ortCoreMLExecutionProviderOptions.createMLProgram = true
    try ortSessionOptions.appendCoreMLExecutionProvider(
      with: ortCoreMLExecutionProviderOptions
    )

    // Create ONNX Runtime session with the model
    self.ortSession = try ORTSession(
      env: ortEnv,
      modelPath: modelUrl.path(percentEncoded: false),
      sessionOptions: ortSessionOptions
    )

    // Validate model input and output names
    let inputNames: [String] = try ortSession.inputNames()
    guard inputNames == [Constants.inputName] else {
      throw SwiftF0DetectorError.incompatibleModelFormat
    }

    let outputNames: [String] = try ortSession.outputNames()
    guard outputNames == Constants.outputNames else {
      throw SwiftF0DetectorError.incompatibleModelFormat
    }
  }

  /// Detects pitch from an audio file at the given URL.
  /// - Parameter url: URL of the audio file to process.
  /// - Returns: Array of PitchResult containing pitch and confidence for each frame.
  /// - Throws: Errors if audio conversion or pitch detection fails.
  public func detect(url: URL) throws -> [PitchResult] {
    // Convert audio file to model input format (Float array)
    var pcmData: [Float] = try convertAudioFileToInput(url: url)

    // Prepare input tensor shape and data
    let shape: [NSNumber] = [1, NSNumber(value: pcmData.count)]
    let byteCount: Int = pcmData.count * MemoryLayout<Float>.size
    let data: NSMutableData = NSMutableData(bytes: &pcmData, length: byteCount)
    let inputValue: ORTValue = try ORTValue(
      tensorData: data,
      elementType: .float,
      shape: shape
    )

    // Run the model with the input tensor
    let outputs: [String: ORTValue] = try ortSession.run(
      withInputs: [Constants.inputName: inputValue],
      outputNames: Set(Constants.outputNames),
      runOptions: nil
    )

    // Extract pitch frequencies from model output
    guard let frequenciesValue: ORTValue = outputs[Constants.outputNames[0]]
    else {
      throw SwiftF0DetectorError.failedToDetectPitch
    }
    let frequenciesRaw: Data = try frequenciesValue.tensorData() as Data
    let frequencies: [Float] = frequenciesRaw.withUnsafeBytes {
      Array($0.bindMemory(to: Float.self))
    }

    // Extract confidence values from model output
    guard let confidencesValue: ORTValue = outputs[Constants.outputNames[1]]
    else {
      throw SwiftF0DetectorError.failedToDetectPitch
    }
    let confidencesRaw: Data = try confidencesValue.tensorData() as Data
    let confidences: [Float] = confidencesRaw.withUnsafeBytes {
      Array($0.bindMemory(to: Float.self))
    }

    // Combine frequencies and confidences into PitchResult array
    let result: [PitchResult] = zip(frequencies, confidences)
      .enumerated()
      .map { (index, value) in
        let (pitchHz, confidence): (Float, Float) = value
        let timeStamp: Float = calculateTimeStamp(frameIndex: index)
        let vocing: Bool = computeVoicing(
          pitchHz: pitchHz,
          confidence: confidence
        )

        return PitchResult(
          pitchHz: pitchHz,
          confidence: confidence,
          timeStamp: timeStamp,
          voicing: vocing
        )
      }

    return result
  }

  // MARK: - Detection Helper Methods

  /// Converts an audio file to a Float array input compatible with the model.
  /// - Parameter url: URL of the audio file.
  /// - Throws: Errors if conversion fails.
  /// - Returns: Array of Float samples at target sample rate and mono channel.
  private func convertAudioFileToInput(url: URL) throws -> [Float] {
    let audioFile: AVAudioFile = try AVAudioFile(forReading: url)

    // Input audio file format and desired output format setup
    let inputFormat: AVAudioFormat = audioFile.processingFormat
    guard
      let outputFormat: AVAudioFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Constants.targetSampleRate,
        channels: 1,
        interleaved: false
      )
    else {
      throw SwiftF0DetectorError.failedToConvertAudioFile
    }

    // Create audio converter from input to output format
    guard
      let converter: AVAudioConverter = AVAudioConverter(
        from: inputFormat,
        to: outputFormat
      )
    else {
      throw SwiftF0DetectorError.failedToConvertAudioFile
    }

    // Read all frames from input audio file into buffer
    let inputFrameCount: AVAudioFrameCount = AVAudioFrameCount(audioFile.length)
    guard
      let inputBuffer: AVAudioPCMBuffer = AVAudioPCMBuffer(
        pcmFormat: inputFormat,
        frameCapacity: inputFrameCount
      )
    else {
      throw SwiftF0DetectorError.failedToConvertAudioFile
    }

    try audioFile.read(into: inputBuffer)

    // Calculate sample rate ratio for conversion
    let sampleRateRatio: Double =
      outputFormat.sampleRate / inputFormat.sampleRate

    // Calculate output frame count with padding to prevent overflow
    let outputFrameCount: AVAudioFrameCount =
      AVAudioFrameCount(Double(inputBuffer.frameLength) * sampleRateRatio)
      + 1024
    guard
      let outputBuffer: AVAudioPCMBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: outputFrameCount
      )
    else {
      throw SwiftF0DetectorError.failedToConvertAudioFile
    }

    // Perform the conversion
    var isConverted: Bool = false
    converter.convert(to: outputBuffer, error: nil) { _, status in
      if isConverted {
        status.pointee = .noDataNow
        return nil
      }

      isConverted = true
      status.pointee = .haveData
      return inputBuffer
    }

    // Extract converted audio data as Float array
    guard
      let channelData: UnsafePointer<UnsafeMutablePointer<Float>> = outputBuffer
        .floatChannelData
    else {
      throw SwiftF0DetectorError.failedToConvertAudioFile
    }

    let result: [Float] = Array(
      UnsafeBufferPointer(
        start: channelData[0],
        count: Int(outputBuffer.frameLength)
      )
    )

    return result
  }

  /// Determines if a pitch is voiced based on frequency and confidence thresholds.
  /// - Parameters:
  ///   - pitchHz: Detected pitch frequency in Hz.
  ///   - confidence: Confidence score of the detection.
  /// - Returns: True if voiced, false otherwise.
  private func computeVoicing(pitchHz: Float, confidence: Float) -> Bool {
    guard pitchHz >= minFrequency else { return false }
    guard pitchHz <= maxFrequency else { return false }
    guard confidence >= confidenceThreshold else { return false }
    return true
  }

  /// Calculates the timestamp for a given frame index.
  /// - Parameter frameIndex: Index of the frame.
  /// - Returns: Timestamp in seconds.
  private func calculateTimeStamp(frameIndex: Int) -> Float {
    return Float(
      (Double(frameIndex * Constants.hopSize) + Constants.centerOffset)
        / Constants.targetSampleRate
    )
  }
}
