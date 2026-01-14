# Icon Generation Instructions

The Electron app requires platform-specific icons for building:

- `icon.icns` - macOS (Apple Icon Image format)
- `icon.ico` - Windows (ICO format)
- `icons/` - Linux (PNG files at various sizes)

## Generating Icons from SVG

### Prerequisites
```bash
# macOS
brew install imagemagick librsvg

# Ubuntu/Debian
sudo apt-get install imagemagick librsvg2-bin

# Or use npm packages
npm install -g png-to-ico
```

### Generate PNGs from SVG
```bash
# Generate various sizes
for size in 16 32 48 64 128 256 512 1024; do
    rsvg-convert -w $size -h $size icon.svg > icons/${size}x${size}.png
done
```

### Generate Windows ICO
```bash
# Using ImageMagick
convert icons/16x16.png icons/32x32.png icons/48x48.png icons/256x256.png icon.ico

# Or using png-to-ico
png-to-ico icons/256x256.png > icon.ico
```

### Generate macOS ICNS
```bash
# On macOS only - create iconset
mkdir icon.iconset
for size in 16 32 64 128 256 512 1024; do
    cp icons/${size}x${size}.png icon.iconset/icon_${size}x${size}.png
done
iconutil -c icns icon.iconset
rm -rf icon.iconset
```

## Quick Development Setup

For development without proper icons, electron-builder will use defaults.
The SVG icon is used for the application UI.

## Production Build

Before creating production builds, generate proper icons using the steps above
to ensure the application displays correctly across all platforms.
