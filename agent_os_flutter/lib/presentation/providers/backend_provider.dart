/// Riverpod providers for backend process state and API client.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/backend_process.dart';
import '../../core/api_client.dart';
import '../../core/database.dart';
import '../../data/repositories/agent_repository.dart';
import '../../data/repositories/local_repository.dart';

/// Exposes the backend process state.
final backendStateProvider = StreamProvider<BackendState>((ref) {
  final backend = ref.watch(backendProcessProvider);
  return backend.stateStream;
});

/// Whether the backend is currently running.
final backendRunningProvider = Provider<bool>((ref) {
  final state = ref.watch(backendStateProvider).valueOrNull;
  return state == BackendState.running;
});

/// Backend error message.
final backendErrorProvider = Provider<String>((ref) {
  final backend = ref.watch(backendProcessProvider);
  return backend.errorMessage;
});

/// API client (depends on backend base URL).
final agentApiClientProvider = Provider<AgentApiClient>((ref) {
  final backend = ref.watch(backendProcessProvider);
  return AgentApiClient(baseUrl: backend.baseUrl);
});

/// Agent repository.
final agentRepositoryProvider = Provider<AgentRepository>((ref) {
  return AgentRepository(ref.watch(agentApiClientProvider));
});

/// Local repository.
final localRepositoryProvider = Provider<LocalRepository>((ref) {
  return LocalRepository(ref.watch(appDatabaseProvider));
});

/// Backend log stream for the log viewer.
final backendLogProvider = StreamProvider<String>((ref) {
  final backend = ref.watch(backendProcessProvider);
  return backend.logStream;
});
