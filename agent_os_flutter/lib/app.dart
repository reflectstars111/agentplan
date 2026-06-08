/// Agent-OS App widget — Material Design 3 theme + routing.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'core/theme.dart';
import 'presentation/screens/home_screen.dart';
import 'presentation/screens/settings_screen.dart';
import 'presentation/screens/trace_screen.dart';

final _router = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(path: '/', builder: (_, __) => const HomeScreen()),
    GoRoute(path: '/settings', builder: (_, __) => const SettingsScreen()),
    GoRoute(path: '/trace/:traceId', builder: (_, state) =>
      TraceScreen(traceId: state.pathParameters['traceId']!)),
  ],
);

class AgentOSApp extends ConsumerWidget {
  const AgentOSApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp.router(
      title: 'Agent-OS Console',
      debugShowCheckedModeBanner: false,
      theme: AgentOSTheme.dark,
      darkTheme: AgentOSTheme.dark,
      themeMode: ThemeMode.dark,
      routerConfig: _router,
    );
  }
}
