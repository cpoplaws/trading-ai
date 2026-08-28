/**
 * Semantic design tokens for the mobile app.
 *
 * These tokens mirror the naming conventions used in web artifacts (index.css)
 * so that multi-artifact projects share a cohesive visual identity.
 *
 * Replace the placeholder values below with values that match the project's
 * brand. If a sibling web artifact exists, read its index.css and convert the
 * HSL values to hex so both artifacts use the same palette.
 *
 * To add dark mode, add a `dark` key with the same token names.
 * The useColors() hook will automatically pick it up.
 */

const colors = {
  light: {
    text: '#f4f7fb',
    tint: '#4d8dff',
    background: '#070b14',
    foreground: '#f4f7fb',
    card: '#0d1422',
    cardForeground: '#f4f7fb',
    primary: '#4d8dff',
    primaryForeground: '#ffffff',
    secondary: '#141d2e',
    secondaryForeground: '#d8e2f2',
    muted: '#182237',
    mutedForeground: '#8290a8',
    accent: '#211d46',
    accentForeground: '#c9c3ff',
    destructive: '#f05d6c',
    destructiveForeground: '#ffffff',
    border: '#1c2940',
    input: '#263550',
    success: '#45d69a',
    successForeground: '#9af0c9',
    warning: '#f4ad62',
  },
  dark: {
    text: '#f4f7fb',
    tint: '#4d8dff',
    background: '#070b14',
    foreground: '#f4f7fb',
    card: '#0d1422',
    cardForeground: '#f4f7fb',
    primary: '#4d8dff',
    primaryForeground: '#ffffff',
    secondary: '#141d2e',
    secondaryForeground: '#d8e2f2',
    muted: '#182237',
    mutedForeground: '#8290a8',
    accent: '#211d46',
    accentForeground: '#c9c3ff',
    destructive: '#f05d6c',
    destructiveForeground: '#ffffff',
    border: '#1c2940',
    input: '#263550',
    success: '#45d69a',
    successForeground: '#9af0c9',
    warning: '#f4ad62',
  },
  radius: 10,
};

export default colors;
