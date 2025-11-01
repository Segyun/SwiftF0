// Copyright Huigyun Jeong. All rights reserved.

/// Container of Note
public struct Note {
  public typealias MIDINoteNumber = UInt8

  /// The position of the note in seconds
  public let position: Double
  /// The pitch of the note represented as a MIDI note number (0-127)
  public let pitch: MIDINoteNumber
  /// The duration of the note in seconds
  public let duration: Double
}
