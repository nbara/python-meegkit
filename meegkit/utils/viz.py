import numpy as np

try:
    import mne
except ImportError:
    mne = None


def plot_montage(chan_list, ax=None, scores=None, title='', cmap=None):
    """Plot montage and color code channels."""
    from mne.viz.utils import _plot_sensors
    from mne.channels.layout import _auto_topomap_coords
    from mne.channels import read_montage
    from mne.utils import check_version
    from matplotlib import cm

    if cmap is None:
        cmap = cm.viridis

    # connectivity, ch_names = read_ch_connectivity(
    #   'biosemi64', picks=montage.selection[:64])
    # plt.imshow(connectivity.toarray(), origin='lower')
    # plt.xlabel('{} Channels'.format(len(ch_names)))
    # plt.ylabel('{} Channels'.format(len(ch_names)))
    # plt.title('Between-sensor adjacency')
    # plt.show()

    montage = read_montage('biosemi64')
    info = mne.create_info(montage.ch_names[:64], sfreq=256, ch_types="eeg",
                           montage='biosemi64')
    pos = _auto_topomap_coords(info, picks=range(64), ignore_overlap=True,
                               to_sphere=True)
    colors = np.ones((64, 4))
    colors[:, 0:3] = 0.9

    if not chan_list or chan_list is None:
        chan_list = montage.ch_names
    picks = mne.pick_channels(montage.ch_names[:64], chan_list)

    if scores is None:
        colors[picks, :] = cmap(100)
    else:
        for ch in chan_list:
            picks2 = mne.pick_channels(montage.ch_names[:64], [ch, ])
            colors[picks2, :] = cmap(scores[ch])

    f = _plot_sensors(pos, colors, [], montage.ch_names[:64], title=title,
                      show_names=False,
                      ax=ax,
                      show=False,
                      select=False,
                      block=False,
                      to_sphere=True)

    # Modify dot size
    scale_factor = 20.
    try:
        ax = ax
    except:  # noqa
        ax = f.axes[0]
    collection = ax.collections[0]
    if check_version("matplotlib", "1.4"):
        collection.set_sizes([scale_factor])
        collection.set_linewidths([0])
    else:
        collection._sizes = [scale_factor]
        collection._linewidths = [0]

    return f
