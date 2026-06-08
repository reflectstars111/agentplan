/// Agent-OS Material Design 3 dark theme.
///
/// Cyberpunk-inspired color scheme: deep navy backgrounds with
/// cyan accents and amber highlights — consistent with the
/// terminal/console aesthetic of the original web UI.
library;

import 'package:flutter/material.dart';

class AgentOSTheme {
  AgentOSTheme._();

  // Brand colors
  static const Color navy = Color(0xFF1A1A2E);
  static const Color darkNavy = Color(0xFF0A0A1A);
  static const Color surface = Color(0xFF16213E);
  static const Color cyan = Color(0xFF00D4FF);
  static const Color amber = Color(0xFFFFB800);
  static const Color green = Color(0xFF00FF88);
  static const Color red = Color(0xFFFF4444);
  static const Color grey = Color(0xFF888888);
  static const Color border = Color(0xFF333355);

  static final ColorScheme _darkColorScheme = ColorScheme.dark(
    primary: cyan,
    secondary: amber,
    surface: surface,
    onPrimary: darkNavy,
    onSecondary: darkNavy,
    onSurface: const Color(0xFFE0E0E0),
    error: red,
    outline: border,
  );

  static final ThemeData dark = ThemeData(
    useMaterial3: true,
    colorScheme: _darkColorScheme,
    scaffoldBackgroundColor: navy,
    appBarTheme: const AppBarTheme(
      backgroundColor: surface,
      foregroundColor: Color(0xFFE0E0E0),
      elevation: 0,
      centerTitle: false,
      titleTextStyle: TextStyle(
        color: cyan,
        fontSize: 18,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.5,
      ),
    ),
    cardTheme: CardThemeData(
      color: surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: const BorderSide(color: border, width: 1),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: darkNavy,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: cyan, width: 2),
      ),
      contentPadding: const EdgeInsets.all(12),
      hintStyle: const TextStyle(color: grey, fontSize: 13),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: const Color(0xFF0F3460),
        foregroundColor: const Color(0xFFE0E0E0),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        textStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: cyan,
        foregroundColor: darkNavy,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        textStyle: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
      ),
    ),
    segmentedButtonTheme: SegmentedButtonThemeData(
      style: SegmentedButton.styleFrom(
        backgroundColor: darkNavy,
        selectedBackgroundColor: cyan,
        selectedForegroundColor: darkNavy,
        foregroundColor: grey,
        side: const BorderSide(color: border),
        textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
      ),
    ),
    chipTheme: ChipThemeData(
      backgroundColor: surface,
      selectedColor: cyan.withAlpha(30),
      labelStyle: const TextStyle(fontSize: 12),
      side: const BorderSide(color: border),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    dividerTheme: const DividerThemeData(
      color: border,
      thickness: 1,
    ),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: surface,
      contentTextStyle: const TextStyle(color: Color(0xFFE0E0E0)),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      behavior: SnackBarBehavior.floating,
    ),
  );
}
