import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROLL = 20   # rolling-average window

# ── Persistent figure (created once, reused every update) ────────────────────
_fig = None
_ax  = None
_ax2 = None

def _init_figure():
    global _fig, _ax, _ax2
    _fig = plt.figure(figsize=(10, 5))
    _fig.patch.set_facecolor('#1e1e2e')
    plt.ion()
    _fig.show()

    gs   = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.35)
    _ax  = _fig.add_subplot(gs[0])
    _ax2 = _fig.add_subplot(gs[1])


def _rolling(data, window):
    out = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out.append(sum(data[start : i + 1]) / (i - start + 1))
    return out


def plot(scores, mean_scores):
    global _fig, _ax, _ax2

    # Create window on first call (or if user closed it)
    if _fig is None or not plt.fignum_exists(_fig.number):
        _init_figure()

    # ── Left panel: training curve ──────────────────────────────────────────
    _ax.cla()
    _ax.set_facecolor('#1e1e2e')
    _ax.tick_params(colors='#cdd6f4')
    for spine in _ax.spines.values():
        spine.set_edgecolor('#45475a')

    rolling = _rolling(scores, ROLL)

    _ax.plot(scores,      color='#89b4fa', linewidth=0.8, alpha=0.45, label='Score')
    _ax.plot(rolling,     color='#a6e3a1', linewidth=2.2,             label=f'Rolling avg ({ROLL}g)')
    _ax.plot(mean_scores, color='#fab387', linewidth=1.0,
             linestyle='--', alpha=0.6, label='All-time mean')

    _ax.set_title('Training Progress', color='#cdd6f4', fontsize=13, pad=10)
    _ax.set_xlabel('Games',  color='#cdd6f4')
    _ax.set_ylabel('Score',  color='#cdd6f4')
    _ax.set_ylim(ymin=0)
    _ax.legend(facecolor='#313244', edgecolor='#45475a',
               labelcolor='#cdd6f4', fontsize=9)

    if scores:
        _ax.annotate(f'{scores[-1]}',
                     xy=(len(scores) - 1, scores[-1]),
                     color='#89b4fa', fontsize=8, ha='left')
        _ax.annotate(f'{rolling[-1]:.0f}',
                     xy=(len(rolling) - 1, rolling[-1]),
                     color='#a6e3a1', fontsize=9, fontweight='bold', ha='left')

    # ── Right panel: stats box ──────────────────────────────────────────────
    _ax2.cla()
    _ax2.set_facecolor('#181825')
    _ax2.axis('off')

    best  = max(scores)       if scores else 0
    last  = scores[-1]        if scores else 0
    r20   = rolling[-1]       if rolling else 0
    games = len(scores)

    stats = [
        ('Games',              f'{games}'),
        ('Last score',         f'{last}'),
        ('Best score',         f'{best}'),
        (f'Avg (last {ROLL})', f'{r20:.0f}'),
        ('All-time avg',       f'{mean_scores[-1]:.0f}' if mean_scores else '0'),
    ]

    _ax2.text(0.5, 0.95, 'Stats', transform=_ax2.transAxes,
              color='#cdd6f4', fontsize=11, fontweight='bold',
              ha='center', va='top')

    for i, (label, value) in enumerate(stats):
        y = 0.80 - i * 0.14
        _ax2.text(0.08, y, label,  transform=_ax2.transAxes, color='#a6adc8', fontsize=9)
        _ax2.text(0.92, y, value,  transform=_ax2.transAxes,
                  color='#cdd6f4', fontsize=10, fontweight='bold', ha='right')

    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.01)
