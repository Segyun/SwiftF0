# SwiftF0: ONNX-based Pitch Detection & Note Conversion Library

SwiftF0 is a simple Swift library that estimates pitch (F0) from audio using an ONNX model and converts the results into musical notes (`Note`). You can detect pitch with `SwiftF0Detector`, check the results with `PitchResult`, and convert them into a sequence of notes using `NoteConverter`.

## Overview of Key Types

- `SwiftF0Detector`
  - Loads an ONNX model and estimates pitch (F0) from audio files/buffers.
  - Initialization: `init(modelUrl: URL)`
  - Usage: `detect(url: URL) throws -> PitchResult`

- `NoteConverter`
  - Analyzes a `PitchResult` to group continuous pitches and calculates note boundaries/durations to generate an array of `Note`.
  - Usage: `convert(_ result: PitchResult) -> [Note]`

## Installation and Requirements

- Compatible with iOS projects (adjust platform/version according to your target)
- Include the model file (`SwiftF0.onnx`) and test audio (`Audio.m4a`) in your app bundle.

## Quick Start: Usage Example

Below is a minimal example demonstrating pitch detection and note conversion using the ONNX model and audio file included in the bundle.

```swift
guard let url = Bundle.main.url(
  forResource: "SwiftF0",
  withExtension: "onnx"
) else {
  return
}

let swiftF0Detector = try SwiftF0Detector(modelUrl: url)

guard let audio = Bundle.main.url(
  forResource: "Audio",
  withExtension: "m4a"
) else {
  print("Audio not found")
  return
}

let result = try swiftF0Detector.detect(url: audio)
print(result)

let notes = NoteConverter.convert(result)
print(notes)
```
