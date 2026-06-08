/// BackendProcess — manages the Python Agent-OS backend as a child process.
///
/// Starts the backend on app launch, monitors health, and stops on app close.
/// Communicates via HTTP REST API on localhost.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Backend lifecycle states.
enum BackendState { stopped, starting, running, error }

/// Manages the Python backend subprocess lifecycle.
class BackendProcess {
  Process? _process;
  final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 2),
    receiveTimeout: const Duration(seconds: 2),
  ));

  BackendState _state = BackendState.stopped;
  BackendState get state => _state;

  String _host = '127.0.0.1';
  int _port = 8000;
  String _errorMessage = '';

  String get host => _host;
  int get port => _port;
  String get errorMessage => _errorMessage;
  String get baseUrl => 'http://$_host:$_port';

  final _stateController = StreamController<BackendState>.broadcast();
  Stream<BackendState> get stateStream => _stateController.stream;

  final _logController = StreamController<String>.broadcast();
  Stream<String> get logStream => _logController.stream;

  /// Start the Python backend as a subprocess.
  Future<void> start({
    String pythonPath = 'python',
    int port = 8000,
    String host = '127.0.0.1',
    String embedMode = 'mock',
    String llmProvider = 'mock',
  }) async {
    if (_state == BackendState.running) return;

    _port = port;
    _host = host;
    _state = BackendState.starting;
    _stateController.add(BackendState.starting);

    try {
      // Resolve the src directory relative to the project root
      final projectRoot = Directory.current.path;

      _logController.add('[Agent-OS] Starting backend: $pythonPath -m src --port $port --embed $embedMode --host $host');

      _process = await Process.start(
        pythonPath,
        ['-m', 'src', '--port', '$port', '--embed', embedMode, '--host', host],
        workingDirectory: projectRoot,
        mode: ProcessStartMode.normal,
      );

      // Pipe stdout/stderr to log stream
      _process!.stdout
          .transform(const Utf8Decoder())
          .transform(const LineSplitter())
          .listen((line) => _logController.add(line));

      _process!.stderr
          .transform(const Utf8Decoder())
          .transform(const LineSplitter())
          .listen((line) {
        _logController.add('[stderr] $line');
        _errorMessage = line;
      });

      _process!.exitCode.then((code) {
        _logController.add('[Agent-OS] Backend exited with code $code');
        _state = BackendState.stopped;
        _stateController.add(BackendState.stopped);
        _process = null;
      });

      // Wait for health check
      final ready = await waitForReady(timeout: const Duration(seconds: 30));
      if (ready) {
        _state = BackendState.running;
        _stateController.add(BackendState.running);
        _logController.add('[Agent-OS] Backend ready on $_host:$_port');
      } else {
        _state = BackendState.error;
        _stateController.add(BackendState.error);
        _errorMessage = 'Backend failed to start within timeout';
        _logController.add('[Agent-OS] $_errorMessage');
      }
    } catch (e) {
      _state = BackendState.error;
      _stateController.add(BackendState.error);
      _errorMessage = 'Failed to start backend: $e';
      _logController.add('[Agent-OS] $_errorMessage');
    }
  }

  /// Poll health endpoint until ready or timeout.
  Future<bool> waitForReady({Duration timeout = const Duration(seconds: 30)}) async {
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      if (await isHealthy()) return true;
      await Future.delayed(const Duration(milliseconds: 500));
    }
    return false;
  }

  /// Check if the backend is responding.
  Future<bool> isHealthy() async {
    try {
      final response = await _dio.get('$baseUrl/health');
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Stop the backend subprocess.
  Future<void> stop() async {
    if (_process != null) {
      _logController.add('[Agent-OS] Stopping backend...');
      _process!.kill(ProcessSignal.sigterm);
      // Force kill after 5 seconds
      Future.delayed(const Duration(seconds: 5), () {
        if (_process != null) {
          _process!.kill(ProcessSignal.sigkill);
        }
      });
      await _process!.exitCode.timeout(
        const Duration(seconds: 10),
        onTimeout: () => -1,
      );
      _process = null;
    }
    _state = BackendState.stopped;
    _stateController.add(BackendState.stopped);
  }

  void dispose() {
    stop();
    _stateController.close();
    _logController.close();
    _dio.close();
  }
}

/// Riverpod provider for the backend process singleton.
final backendProcessProvider = Provider<BackendProcess>((ref) {
  throw UnimplementedError('Must be overridden in main()');
});
