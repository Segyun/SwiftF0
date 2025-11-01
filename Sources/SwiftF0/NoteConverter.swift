// Copyright Huigyun Jeong. All rights reserved.

import Foundation

/// A class responsible for converting pitch detection results into musical notes.
public final class NoteConverter {
  /// A structure representing a continuous segment of pitch data.
  /// Each segment contains a start time, end time, and a collection of pitch results.
  private struct Segment {
    let startTime: Float
    var endTime: Float
    var pitchResults: [Float]
  }

  private init() {}

  /**
   Converts an array of pitch detection results into an array of musical notes.
  
   - Parameters:
     - pitchResults: An array of `PitchResult` containing pitch frequency and confidence.
     - unvoiceGracePeriod: The allowed duration of unvoiced frames before ending a segment (default 0.02 seconds).
     - minNoteDuration: The minimum duration for a segment to be considered a valid note (default 0.05 seconds).
     - semitoneThreshold: The maximum allowed pitch difference in semitones within a segment (default 0.8).
  
   - Returns: An array of `Note` objects representing the detected musical notes.
   */
  public static func convert(
    _ pitchResults: [PitchResult],
    unvoiceGracePeriod: Float = 0.02,
    minNoteDuration: Float = 0.05,
    semitoneThreshold: Float = 0.8,
  ) -> [Note] {
    let frameDuration: Float = 0.016

    var segments: [Segment] = []
    var currentSegment: Segment? = nil
    var consecutiveUnvoicedFrames: Int = 0

    for (index, pitchResult) in pitchResults.enumerated() {
      let pitchHz: Float = pitchResult.pitchHz
      let voicing: Bool = pitchResult.voicing

      let currentTime: Float = Float(index) * frameDuration
      let pitch: Float = convertToPitch(frequency: pitchHz)

      if voicing {
        consecutiveUnvoicedFrames = 0

        if let segment = currentSegment {
          let medianPitch: Float = median(of: segment.pitchResults) ?? 0.0
          let semitone: Float = abs(medianPitch - pitch)

          if semitone > semitoneThreshold {
            segments.append(segment)
            currentSegment = Segment(
              startTime: currentTime,
              endTime: currentTime + frameDuration,
              pitchResults: [pitch]
            )
          } else {
            currentSegment?.pitchResults.append(pitch)
            currentSegment?.endTime += frameDuration
          }
        } else {
          currentSegment = Segment(
            startTime: currentTime,
            endTime: currentTime + frameDuration,
            pitchResults: [pitch]
          )
        }
      } else {
        if let segment = currentSegment {
          consecutiveUnvoicedFrames += 1
          let unvoicedDuration: Float =
            Float(consecutiveUnvoicedFrames) * frameDuration

          if unvoicedDuration >= unvoiceGracePeriod {
            segments.append(segment)
            currentSegment = nil
            consecutiveUnvoicedFrames = 0
          } else {
            currentSegment?.endTime = currentTime + frameDuration
          }
        }
      }
    }

    if let currentSegment {
      segments.append(currentSegment)
    }

    if segments.isEmpty {
      return []
    }

    let filteredNotes: [Note] = segments.compactMap { segment in
      let duration: Float = segment.endTime - segment.startTime

      if duration >= minNoteDuration {
        let medianPitch: Int = Int(
          round(median(of: segment.pitchResults) ?? 0.0)
        )

        return Note(
          position: Double(segment.startTime),
          pitch: Note.MIDINoteNumber(medianPitch),
          duration: Double(segment.endTime - segment.startTime)
        )
      }
      return nil
    }

    if filteredNotes.isEmpty {
      return []
    }

    var finalNotes: [Note] = [filteredNotes[0]]
    let timeTolerance: Float = 1e-9

    for note in filteredNotes[1..<filteredNotes.endIndex] {
      guard let previousNote = finalNotes.last else { continue }
      let timeGap =
        note.position - (previousNote.position + previousNote.duration)

      if timeGap <= Double(frameDuration + timeTolerance),
        previousNote.pitch == note.pitch
      {
        let updatedNote: Note = Note(
          position: previousNote.position,
          pitch: previousNote.pitch,
          duration: previousNote.duration + note.duration
        )
        finalNotes[finalNotes.endIndex - 1] = updatedNote
      } else {
        finalNotes.append(note)
      }
    }

    return finalNotes
  }

  /**
   Converts a frequency in Hertz to a MIDI pitch number.
  
   - Parameter frequency: The frequency in Hertz.
   - Returns: The corresponding MIDI pitch number as a Float. Returns 0 if frequency is not positive.
   */
  private static func convertToPitch(frequency: Float) -> Float {
    guard frequency > 0 else { return 0 }
    let midi: Float = 69 + 12 * log2(Float(frequency) / 440.0)
    return midi
  }

  /**
   Calculates the median value from an array of Float numbers.
  
   - Parameter numbers: An array of Float values.
   - Returns: The median value as a Float, or nil if the array is empty.
   */
  private static func median(of numbers: [Float]) -> Float? {
    guard !numbers.isEmpty else { return nil }

    let sorted: [Float] = numbers.sorted()
    let count: Int = sorted.count

    if count % 2 == 1 {
      return sorted[count / 2]
    } else {
      let mid1 = sorted[count / 2 - 1]
      let mid2 = sorted[count / 2]
      return (mid1 + mid2) / 2.0
    }
  }
}
