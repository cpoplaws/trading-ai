import { Feather } from '@expo/vector-icons';
import { StyleSheet, Text, View } from 'react-native';
import { Card, EmptyState, ErrorState, formatCurrency, Header, LoadingState, Screen, SectionHeading, StatusPill } from '@/components/TradingUI';
import { useTradesQuery } from '@/lib/trading-api';
import { useColors } from '@/hooks/useColors';

export default function TradesScreen() {
  const colors = useColors();
  const query = useTradesQuery();
  const trades = query.data ?? [];

  return (
    <Screen refreshing={query.isRefetching} onRefresh={() => void query.refetch()}>
      <Header eyebrow="QUANTLYTICS / ACTIVITY" title="Recent trades" subtitle="A live audit trail of your autonomous execution." action={<StatusPill active label="LIVE FEED" />} />
      <SectionHeading title="Execution log" action={trades.length ? `${trades.length} events` : undefined} />
      {query.isLoading ? <Card><LoadingState label="Loading trade activity…" /></Card> : query.isError ? <ErrorState message={query.error instanceof Error ? query.error.message : 'Trade activity is unavailable.'} onRetry={() => void query.refetch()} /> : trades.length === 0 ? <EmptyState icon="swap-vertical-outline" title="No trades yet" message="Enable a strategy to start building your execution history." /> : (
        <View style={[styles.list, { backgroundColor: colors.card, borderColor: colors.border }]}>
          {trades.map((trade, index) => {
            const isBuy = trade.side.toLowerCase() === 'buy';
            const pnlPositive = (trade.pnl ?? 0) >= 0;
            return (
              <View key={trade.id} style={[styles.row, index < trades.length - 1 && { borderBottomColor: colors.border, borderBottomWidth: 1 }]}>
                <View style={[styles.side, { backgroundColor: isBuy ? colors.accent : colors.secondary }]}><Feather name={isBuy ? 'arrow-down-left' : 'arrow-up-right'} size={15} color={isBuy ? colors.success : colors.destructive} /></View>
                <View style={styles.tradeCopy}><View style={styles.symbolRow}><Text style={[styles.symbol, { color: colors.foreground }]}>{trade.symbol}</Text><Text style={[styles.quantity, { color: colors.mutedForeground }]}>×{trade.quantity}</Text></View><Text style={[styles.tradeMeta, { color: colors.mutedForeground }]}>{trade.strategy} · {new Date(trade.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</Text></View>
                <View style={styles.tradeRight}><Text style={[styles.price, { color: colors.foreground }]}>{formatCurrency(trade.price)}</Text>{trade.pnl !== undefined ? <Text style={[styles.pnl, { color: pnlPositive ? colors.success : colors.destructive }]}>{pnlPositive ? '+' : ''}{formatCurrency(trade.pnl)}</Text> : null}</View>
              </View>
            );
          })}
        </View>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  list: { marginHorizontal: 20, borderWidth: 1, borderRadius: 14, paddingHorizontal: 14 },
  row: { minHeight: 70, paddingVertical: 13, flexDirection: 'row', alignItems: 'center', gap: 11 },
  side: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  tradeCopy: { flex: 1 },
  symbolRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  symbol: { fontFamily: 'Inter_600SemiBold', fontSize: 13 },
  quantity: { fontFamily: 'Inter_400Regular', fontSize: 11 },
  tradeMeta: { fontFamily: 'Inter_400Regular', fontSize: 10, marginTop: 5 },
  tradeRight: { alignItems: 'flex-end' },
  price: { fontFamily: 'Inter_600SemiBold', fontSize: 12 },
  pnl: { fontFamily: 'Inter_600SemiBold', fontSize: 11, marginTop: 5 },
});