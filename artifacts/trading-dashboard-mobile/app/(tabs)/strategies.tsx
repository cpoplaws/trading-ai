import { Feather } from '@expo/vector-icons';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Card, ErrorState, formatCurrency, Header, LoadingState, Screen, SectionHeading, StatusPill, Toggle } from '@/components/TradingUI';
import { useStrategiesQuery, useStrategyToggleMutation } from '@/lib/trading-api';
import { useColors } from '@/hooks/useColors';

export default function StrategiesScreen() {
  const colors = useColors();
  const query = useStrategiesQuery();
  const mutation = useStrategyToggleMutation();
  const [pendingId, setPendingId] = useState<string | null>(null);

  const toggle = async (strategyId: string, enabled: boolean) => {
    setPendingId(strategyId);
    try { await mutation.mutateAsync({ strategyId, enabled }); } finally { setPendingId(null); }
  };

  return (
    <Screen refreshing={query.isRefetching} onRefresh={() => void query.refetch()}>
      <Header eyebrow="QUANTLYTICS / EXECUTION" title="Strategies" subtitle="Enable the edges that fit your current market thesis." action={<StatusPill active label="SYNCED" />} />
      <SectionHeading title="Strategy grid" action={`${query.data?.filter((item) => item.enabled).length ?? 0} active`} />
      {query.isLoading ? <Card><LoadingState label="Loading strategies…" /></Card> : query.isError ? <ErrorState message={query.error instanceof Error ? query.error.message : 'Strategies are unavailable.'} onRetry={() => void query.refetch()} /> : query.data?.length ? (
        <View style={styles.grid}>
          {query.data.map((strategy) => {
            const positive = strategy.pnl >= 0;
            return (
              <Card key={strategy.id} style={styles.strategyCard}>
                <View style={styles.cardTop}>
                  <View style={[styles.strategyIcon, { backgroundColor: strategy.enabled ? colors.accent : colors.secondary }]}>
                    <Feather name={strategy.enabled ? 'activity' : 'pause'} size={16} color={strategy.enabled ? colors.primary : colors.mutedForeground} />
                  </View>
                  <Toggle enabled={strategy.enabled} loading={pendingId === strategy.id} label={`${strategy.name} strategy`} testID={`strategy-toggle-${strategy.id}`} onPress={() => void toggle(strategy.id, !strategy.enabled)} />
                </View>
                <Text style={[styles.strategyName, { color: colors.foreground }]} numberOfLines={2}>{strategy.name}</Text>
                <Text style={[styles.pnl, { color: positive ? colors.success : colors.destructive }]}>{positive ? '+' : ''}{formatCurrency(strategy.pnl)}</Text>
                <View style={styles.detailRow}><Text style={[styles.detailLabel, { color: colors.mutedForeground }]}>Trades</Text><Text style={[styles.detailValue, { color: colors.foreground }]}>{strategy.trades}</Text></View>
                <View style={styles.detailRow}><Text style={[styles.detailLabel, { color: colors.mutedForeground }]}>Win rate</Text><Text style={[styles.detailValue, { color: colors.foreground }]}>{(strategy.win_rate * 100).toFixed(1)}%</Text></View>
                <Pressable accessibilityRole="button" onPress={() => void toggle(strategy.id, !strategy.enabled)} style={({ pressed }) => [styles.action, { borderTopColor: colors.border, opacity: pressed ? 0.65 : 1 }]}>
                  <Text style={[styles.actionText, { color: strategy.enabled ? colors.destructive : colors.primary }]}>{strategy.enabled ? 'Disable strategy' : 'Enable strategy'}</Text>
                  <Feather name="arrow-up-right" size={14} color={strategy.enabled ? colors.destructive : colors.primary} />
                </Pressable>
              </Card>
            );
          })}
        </View>
      ) : <EmptyStrategies />}
    </Screen>
  );
}

function EmptyStrategies() {
  const colors = useColors();
  return <View style={[styles.empty, { backgroundColor: colors.card, borderColor: colors.border }]}><Feather name="layers" size={22} color={colors.mutedForeground} /><Text style={[styles.emptyTitle, { color: colors.foreground }]}>No strategies returned</Text><Text style={[styles.emptyText, { color: colors.mutedForeground }]}>Connect the trading backend to manage your strategy catalog.</Text></View>;
}

const styles = StyleSheet.create({
  grid: { paddingHorizontal: 20, flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: 12 },
  strategyCard: { width: '48.2%', padding: 14 },
  cardTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  strategyIcon: { width: 32, height: 32, borderRadius: 11, alignItems: 'center', justifyContent: 'center' },
  strategyName: { fontFamily: 'Inter_600SemiBold', fontSize: 13, lineHeight: 18, marginTop: 14, minHeight: 36 },
  pnl: { fontFamily: 'Inter_700Bold', fontSize: 19, marginTop: 12, letterSpacing: -0.4 },
  detailRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 9 },
  detailLabel: { fontFamily: 'Inter_400Regular', fontSize: 10 },
  detailValue: { fontFamily: 'Inter_600SemiBold', fontSize: 11 },
  action: { borderTopWidth: 1, marginTop: 14, paddingTop: 12, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  actionText: { fontFamily: 'Inter_600SemiBold', fontSize: 10 },
  empty: { marginHorizontal: 20, borderWidth: 1, borderRadius: 14, padding: 28, alignItems: 'center' },
  emptyTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 15, marginTop: 12 },
  emptyText: { fontFamily: 'Inter_400Regular', fontSize: 12, lineHeight: 18, textAlign: 'center', marginTop: 6 },
});