/// Agent-OS Flutter Desktop App — main entry point.
///
/// Starts the Python backend as a subprocess, initializes the local
/// SQLite database, and launches the Material Design 3 UI.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:window_manager/window_manager.dart';
import 'app.dart';
import 'core/backend_process.dart';
import 'core/database.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Configure window
  await windowManager.ensureInitialized();
  await windowManager.setTitle('Agent-OS Console');
  await windowManager.setSize(const Size(1200, 800));
  await windowManager.setMinimumSize(const Size(900, 600));
  await windowManager.center();
  await windowManager.show();

  // Initialize local database
  final db = AppDatabase();
  await db.initialize();

  // Start Python backend
  final backend = BackendProcess();
  backend.start(port: 8000);

  runApp(
    ProviderScope(
      overrides: [
        backendProcessProvider.overrideWithValue(backend),
        appDatabaseProvider.overrideWithValue(db),
      ],
      child: const AgentOSApp(),
    ),
  );
}
