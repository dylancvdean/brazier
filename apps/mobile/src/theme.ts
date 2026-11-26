import { createContext, useContext } from 'react';

export type ThemePreference = 'system' | 'dark' | 'light';
export type Palette = {
  dark: boolean; bg: string; panel: string; raised: string; line: string; muted: string; soft: string;
  text: string; accent: string; accentSoft: string; onAccent: string; danger: string; dangerSoft: string;
};

// The dark palette is the desktop app's: true black with small surface steps and the ember accent.
export const palettes: { dark: Palette; light: Palette } = {
  dark: {
    dark: true, bg: '#000000', panel: '#0b0b0c', raised: '#131315', line: '#1e1e20', muted: '#7c7c82',
    soft: '#c8c8cd', text: '#ededef', accent: '#ff8a3d', accentSoft: '#2a1608', onAccent: '#1a0e04',
    danger: '#ff7a6b', dangerSoft: '#2a0f0b',
  },
  light: {
    dark: false, bg: '#ffffff', panel: '#f6f6f7', raised: '#ececee', line: '#dcdce0', muted: '#6d6d74',
    soft: '#3a3a40', text: '#17171a', accent: '#d9601a', accentSoft: '#fdebdd', onAccent: '#ffffff',
    danger: '#b3261e', dangerSoft: '#fbe9e7',
  },
};

export function resolvePalette(preference: ThemePreference, system: string | null | undefined): Palette {
  const scheme = preference === 'system' ? (system === 'light' ? 'light' : 'dark') : preference;
  return palettes[scheme];
}

export const ThemeContext = createContext<Palette>(palettes.dark);
export const useTheme = () => useContext(ThemeContext);
