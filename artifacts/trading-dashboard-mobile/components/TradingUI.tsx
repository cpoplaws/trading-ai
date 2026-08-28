import { Feather, Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { ReactNode } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleProp,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import { useColors } from '@/hooks/useColors';

export const AGENT_INFO: Record<string, { name: string; description: string; icon: keyof typeof Feather.glyphMap; color: 'blue' | 'violet' | 'green' | 'orange' }> = {
  execution: { name: 'Execution', description: 'Timing and sizing', icon: 'zap', color: 'blue' },
  risk: { name: 'Risk', description: 'Portfolio guardrails', icon: 'shield', color: 'orange' },
  arbitrage: { name: 'Arbitrage', description: 'Price discrepancies', icon: 'repeat', color: 'violet' },
  market_making: { name: 'Market making', description: 'Liquidity and spread', icon: 'bar-chart-2', color: 'green' },
};

export function formatCurrency(value: number | undefined, compact = false) {
  const amount = value ?? 0;
  return amount.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 1 : 2,
  });
}

export function Screen({ children, refreshing = false, onRefresh }: { children: ReactNode; refreshing?: boolean; onRefresh?: () => void }) {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const webTopInset = Platform.OS === 'web' ? 67 : 0;
  const webBottomInset = Platform.OS === 'web' ? 34 : 0;

  return (
    <View style={[styles.screen, { backgroundColor: colors.background }]}>
      <ScrollView
        contentContainerStyle={{ paddingTop: insets.top + webTopInset, paddingBottom: insets.bottom + webBottomInset + 104 }}
        refreshControl={onRefresh ? <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.primary} /> : undefined}
        showsVerticalScrollIndicator={false}
      >
        {children}
      </ScrollView>
    </View>
  );
}

export function Header({ eyebrow, title, subtitle, action }: { eyebrow: string; title: string; subtitle?: string; action?: ReactNode }) {
  const colors = useColors();
  return (
    <View style={styles.header}>
      <View style={styles.headerCopy}>
        <Text style={[styles.eyebrow, { color: colors.primary }]}>{eyebrow}</Text>
        <Text style={[styles.title, { color: colors.foreground }]}>{title}</Text>
        {subtitle ? <Text style={[styles.subtitle, { color: colors.mutedForeground }]}>{subtitle}</Text> : null}
      </View>
      {action}
    </View>
  );
}

export function IconButton({ icon, onPress, label, testID }: { icon: keyof typeof Ionicons.glyphMap; onPress?: () => void; label: string; testID?: string }) {
  const colors = useColors();
  return (
    <Pressable
      accessibilityLabel={label}
      testID={testID}
      onPress={onPress}
      style={({ pressed }) => [styles.iconButton, { backgroundColor: colors.card, borderColor: colors.border, opacity: pressed ? 0.72 : 1 }]}
    >
      <Ionicons name={icon} size={18} color={colors.foreground} />
    </Pressable>
  );
}

export function SectionHeading({ title, action, onAction }: { title: string; action?: string; onAction?: () => void }) {
  const colors = useColors();
  return (
    <View style={styles.sectionHeading}>
      <Text style={[styles.sectionTitle, { color: colors.foreground }]}>{title}</Text>
      {action ? (
        <Pressable accessibilityRole="button" onPress={onAction} hitSlop={12}>
          <Text style={[styles.sectionAction, { color: colors.primary }]}>{action}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

export function StatusPill({ active, label }: { active: boolean; label: string }) {
  const colors = useColors();
  return (
    <View style={[styles.statusPill, { backgroundColor: active ? colors.accent : colors.secondary }]}>
      <View style={[styles.statusDot, { backgroundColor: active ? colors.success : colors.mutedForeground }]} />
      <Text style={[styles.statusText, { color: active ? colors.successForeground : colors.mutedForeground }]}>{label}</Text>
    </View>
  );
}

export function Toggle({ enabled, onPress, label, loading = false, testID }: { enabled: boolean; onPress: () => void; label: string; loading?: boolean; testID?: string }) {
  const colors = useColors();
  return (
    <Pressable
      accessibilityRole="switch"
      accessibilityState={{ checked: enabled, disabled: loading }}
      accessibilityLabel={label}
      testID={testID}
      onPress={() => {
        void Haptics.selectionAsync();
        onPress();
      }}
      disabled={loading}
      style={({ pressed }) => [{ opacity: pressed ? 0.76 : loading ? 0.5 : 1 }]}
    >
      <View style={[styles.toggleTrack, { backgroundColor: enabled ? colors.primary : colors.muted, borderColor: enabled ? colors.primary : colors.border }]}>
        {loading ? <ActivityIndicator size="small" color={colors.foreground} /> : <View style={[styles.toggleThumb, { backgroundColor: enabled ? colors.primaryForeground : colors.mutedForeground, transform: [{ translateX: enabled ? 10 : -10 }] }]} />}
      </View>
    </Pressable>
  );
}

export function LoadingState({ label = 'Syncing live data…' }: { label?: string }) {
  const colors = useColors();
  return (
    <View style={styles.centerState}>
      <ActivityIndicator color={colors.primary} />
      <Text style={[styles.stateText, { color: colors.mutedForeground }]}>{label}</Text>
    </View>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  const colors = useColors();
  return (
    <View style={[styles.errorState, { backgroundColor: colors.card, borderColor: colors.border }]}>
      <Ionicons name="cloud-offline-outline" size={22} color={colors.destructive} />
      <Text style={[styles.errorTitle, { color: colors.foreground }]}>Couldn’t reach the backend</Text>
      <Text style={[styles.errorMessage, { color: colors.mutedForeground }]}>{message}</Text>
      <Pressable onPress={onRetry} style={({ pressed }) => [styles.retryButton, { backgroundColor: colors.secondary, opacity: pressed ? 0.7 : 1 }]}>
        <Text style={[styles.retryText, { color: colors.primary }]}>Try again</Text>
      </Pressable>
    </View>
  );
}

export function EmptyState({ icon, title, message }: { icon: keyof typeof Ionicons.glyphMap; title: string; message: string }) {
  const colors = useColors();
  return (
    <View style={[styles.emptyState, { backgroundColor: colors.card, borderColor: colors.border }]}>
      <View style={[styles.emptyIcon, { backgroundColor: colors.secondary }]}>
        <Ionicons name={icon} size={21} color={colors.mutedForeground} />
      </View>
      <Text style={[styles.emptyTitle, { color: colors.foreground }]}>{title}</Text>
      <Text style={[styles.emptyMessage, { color: colors.mutedForeground }]}>{message}</Text>
    </View>
  );
}

export function QuickLink({ icon, label, value, color, onPress }: { icon: keyof typeof Ionicons.glyphMap; label: string; value: string; color: string; onPress: () => void }) {
  const colors = useColors();
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.quickLink, { backgroundColor: colors.card, borderColor: colors.border, opacity: pressed ? 0.78 : 1 }]}>
      <View style={[styles.quickIcon, { backgroundColor: color }]}>
        <Ionicons name={icon} size={18} color={colors.foreground} />
      </View>
      <View style={styles.quickCopy}>
        <Text style={[styles.quickLabel, { color: colors.mutedForeground }]}>{label}</Text>
        <Text style={[styles.quickValue, { color: colors.foreground }]}>{value}</Text>
      </View>
      <Ionicons name="chevron-forward" size={16} color={colors.mutedForeground} />
    </Pressable>
  );
}

export function Metric({ label, value, hint, positive }: { label: string; value: string; hint: string; positive?: boolean }) {
  const colors = useColors();
  return (
    <View style={[styles.metric, { backgroundColor: colors.card, borderColor: colors.border }]}>
      <Text style={[styles.metricLabel, { color: colors.mutedForeground }]}>{label}</Text>
      <Text style={[styles.metricValue, { color: colors.foreground }]}>{value}</Text>
      <Text style={[styles.metricHint, { color: positive === undefined ? colors.mutedForeground : positive ? colors.success : colors.destructive }]}>{hint}</Text>
    </View>
  );
}

export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  const colors = useColors();
  return <View style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border, borderRadius: colors.radius }, style]}>{children}</View>;
}

export function useTabNavigate() {
  const router = useRouter();
  return (tab: 'index' | 'strategies' | 'agents' | 'trades') => router.push(`/(tabs)/${tab}` as never);
}

export const styles = StyleSheet.create({
  screen: { flex: 1 },
  header: { paddingHorizontal: 20, paddingBottom: 24, flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' },
  headerCopy: { flex: 1, paddingRight: 14 },
  eyebrow: { fontFamily: 'Inter_700Bold', fontSize: 11, letterSpacing: 1.7, marginBottom: 8 },
  title: { fontFamily: 'Inter_700Bold', fontSize: 29, letterSpacing: -0.8 },
  subtitle: { fontFamily: 'Inter_400Regular', fontSize: 13, marginTop: 8, lineHeight: 19 },
  iconButton: { width: 40, height: 40, borderWidth: 1, borderRadius: 13, alignItems: 'center', justifyContent: 'center' },
  sectionHeading: { paddingHorizontal: 20, marginBottom: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  sectionTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 17, letterSpacing: -0.2 },
  sectionAction: { fontFamily: 'Inter_600SemiBold', fontSize: 12 },
  statusPill: { flexDirection: 'row', alignItems: 'center', gap: 7, paddingHorizontal: 10, paddingVertical: 7, borderRadius: 20 },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusText: { fontFamily: 'Inter_700Bold', fontSize: 10, letterSpacing: 0.7 },
  card: { marginHorizontal: 20, borderWidth: 1, padding: 16 },
  centerState: { minHeight: 160, alignItems: 'center', justifyContent: 'center', gap: 12 },
  stateText: { fontFamily: 'Inter_400Regular', fontSize: 13 },
  errorState: { marginHorizontal: 20, padding: 22, borderWidth: 1, borderRadius: 14, alignItems: 'center' },
  errorTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 15, marginTop: 12 },
  errorMessage: { fontFamily: 'Inter_400Regular', fontSize: 12, textAlign: 'center', lineHeight: 18, marginTop: 6 },
  retryButton: { borderRadius: 9, paddingHorizontal: 16, paddingVertical: 10, marginTop: 16 },
  retryText: { fontFamily: 'Inter_600SemiBold', fontSize: 12 },
  emptyState: { marginHorizontal: 20, padding: 24, borderWidth: 1, borderRadius: 14, alignItems: 'center' },
  emptyIcon: { width: 44, height: 44, borderRadius: 15, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  emptyTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 15 },
  emptyMessage: { fontFamily: 'Inter_400Regular', fontSize: 12, textAlign: 'center', lineHeight: 18, marginTop: 6, maxWidth: 280 },
  toggleTrack: { width: 38, height: 24, borderRadius: 14, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  toggleThumb: { width: 16, height: 16, borderRadius: 8 },
  quickLink: { flex: 1, minWidth: 145, borderWidth: 1, borderRadius: 14, padding: 13, flexDirection: 'row', alignItems: 'center', gap: 10 },
  quickIcon: { width: 34, height: 34, borderRadius: 11, alignItems: 'center', justifyContent: 'center' },
  quickCopy: { flex: 1 },
  quickLabel: { fontFamily: 'Inter_400Regular', fontSize: 11 },
  quickValue: { fontFamily: 'Inter_600SemiBold', fontSize: 13, marginTop: 3 },
  metric: { width: '48%', borderWidth: 1, borderRadius: 13, padding: 14, marginBottom: 10 },
  metricLabel: { fontFamily: 'Inter_500Medium', fontSize: 11 },
  metricValue: { fontFamily: 'Inter_700Bold', fontSize: 19, marginTop: 8, letterSpacing: -0.4 },
  metricHint: { fontFamily: 'Inter_400Regular', fontSize: 10, marginTop: 5 },
});