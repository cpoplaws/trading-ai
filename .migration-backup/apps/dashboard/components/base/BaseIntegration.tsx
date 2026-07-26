'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export function BaseIntegration() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Base Network</CardTitle>
        <CardDescription>Trade on Coinbase's L2</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="text-sm text-muted-foreground">
            <p>⚡ Fast: ~2 second blocks</p>
            <p>💰 Cheap: ~$0.01 per trade</p>
            <p>🔒 Secure: Ethereum L1 backed</p>
          </div>
          <Button className="w-full" disabled>
            Connect Wallet (Coming Soon)
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
