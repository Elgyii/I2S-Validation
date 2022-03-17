from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import ticker, pyplot as plt
from scipy import stats


def correl(xi, yi):
    xi = np.asarray(xi)
    yi = np.asarray(yi)

    f = np.poly1d(np.polyfit(xi, yi, 1))
    m, b = f.coeffs

    # x = np.log10(np.logspace(xi.min(), xi.max(), 100))
    # y = x * m + b

    x = np.logspace(np.min(xi), np.max(xi), 100)
    y = (x ** m) * (10 ** b)

    r, p = stats.spearmanr(xi, yi)
    p = 0.001 if p < 0.001 else (0.01 if p < 0.01 else (0.05 if p < 0.05 else p))

    # bias = np.sum(yi - xi) / xi.size
    bias = np.power(10, np.sum(yi - xi) / xi.size)
    m, b = f.coeffs
    if b > 0:
        b = f'+ {b:.3f}'
    else:
        b = f'{b:.3f}'

    # predict y values of origional data using the fit
    p_y = f(xi)
    # calculate the y-error (residuals)
    yp = yi - p_y
    # sum of the squares of the residuals | SSE
    sse = np.sum(np.power(yp, 2))
    # sum of the squares total | SST
    sst = np.sum(np.power(yi - yi.mean(), 2))
    # r-squared
    rsq = 1 - sse / sst

    base = '_{10}'
    txt = fr'$\log {base} (y)={m:.3f} \\times \log {base} (x) {b}$\n' \
          fr'$N={xi.size}$\n$R^{2}={rsq:.2f}$\n$r={r:.2f}$\n' \
          fr'$p<{p:.3f}$\n$\delta={bias:.2f}$'
    return x, y, txt


def custom_plt(plt):
    plt.rcParams.update({
        'font.size': 30,
        # 'axes.grid': True,
        'axes.linewidth': 2,
        'grid.linewidth': 1,

        'xtick.major.size': 8,
        'xtick.major.width': 2,

        'xtick.minor.visible': True,
        'xtick.minor.size': 4,
        'xtick.minor.width': 1,

        'ytick.major.size': 8,
        'ytick.major.width': 2,

        'ytick.minor.visible': True,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1,

        'savefig.facecolor': '0.8'
    })
    return


def log_ticks(ax, n, axes: str = 'both'):
    # set y ticks
    subs = np.hstack((np.arange(2, 10) * 10 ** -2,
                      np.arange(2, 10) * 10 ** -1,
                      np.arange(2, 10) * 10 ** 0,
                      np.arange(2, 10) * 10 ** 1))
    major = ticker.LogLocator(base=10, numticks=5)
    minor = ticker.LogLocator(base=10, subs=subs, numticks=10)

    if axes == 'both':
        ax.xaxis.set_major_locator(major)
        ax.yaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_xticks())])
        ax.set_yticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_yticks())])

    if axes == 'x':
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_xticks())])
    if axes == 'y':
        ax.yaxis.set_major_locator(major)
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_yticks())])
    return


def overestimated(xdata, ydata, ratio):
    idx = np.where(ydata > xdata * ratio)
    return xdata[idx], ydata[idx]


def underestimated(xdata, ydata, ratio):
    idx = np.where(xdata > ydata * ratio)
    return xdata[idx], ydata[idx]


def fmt_meta(meta: list, ncols: int = 3):
    mn, count, m = len(meta), 0, []
    ml = max([len(f'{s}') for s in meta]) + 2
    while count < mn:
        control = 0
        tmp = []
        for j in range(1, ncols + 1):
            if control == ncols:
                break
            if len(meta) == 0:
                break
            control += 1
            count += 1
            tmp.append(f'{count:02}: {meta.pop(0):<{ml}}')
        if len(tmp) == 0:
            break
        m.append(' | '.join(tmp))

    return '\n\t'.join(m)


def get_under_over(x, y, ratio, case: str = 'over'):
    if case == 'over':
        return overestimated(
            xdata=x
            , ydata=y
            , ratio=ratio)
    else:
        return underestimated(
            xdata=x
            , ydata=y
            , ratio=ratio)


def get_scatter(xdata
                , ydata
                , ax
                , ratio: list
                , alpha: float
                , fill: str
                , over: bool = False
                , lw: float = 2
                , sen: str = None
                , under: bool = False
                , color: str = 'k'
                , marker: str = 'o'
                , xlim: list = None
                , ylim: list = None
                , label: str = None
                , s: int = 150
                , xtick: list = None
                , ytick: list = None):
    if ytick is None:
        ytick = [0.01, 0.1, 1, 10, 100]
    if xtick is None:
        xtick = [0.01, 0.1, 1, 10, 100]
    if ylim is None:
        ylim = [0.009, 10 ** 2]
    if xlim is None:
        xlim = [0.009, 10 ** 2]

    unique = np.unique(xdata['Year'])
    # print(unique)
    markers = {
        '2009': {'m': 'o', 'c': 'k', 'a': 1, 'f': 'w'}
        , '2010': {'m': 'o', 'c': 'k', 'a': 0.5, 'f': 'k'}
        , '2011': {'m': '.', 'c': 'k', 'a': 1, 'f': 'k'}
        , '2012': {'m': 'd', 'c': 'k', 'a': 0.5, 'f': 'k'}
        , '2013': {'m': 'd', 'c': 'k', 'a': 1, 'f': 'w'}
        , '2014': {'m': 'x', 'c': 'k', 'a': 0.5, 'f': 'k'}
        , '2015': {'m': 'X', 'c': 'k', 'a': 1, 'f': 'w'}
        , '2016': {'m': '*', 'c': 'k', 'a': 1, 'f': 'w'}
        , '2017': {'m': '>', 'c': 'k', 'a': 1, 'f': 'w'}
        , '2018': {'m': 'h', 'c': 'k', 'a': 0.5, 'f': 'k'}

    }

    for i, year in enumerate(unique):
        idx = xdata['Year'].isin([year])
        xvals = xdata.loc[idx, 'x'].values
        yvals = ydata.loc[idx].values

        n = ydata.loc[idx].dropna().size
        if n > 0:
            lw = 2.5 if fill == 'w' else 1.5
            ax.scatter(
                xvals,
                yvals,
                # c=color,
                marker=markers[year]['m'],
                facecolors=markers[year]['f'],
                edgecolors=markers[year]['c'],
                alpha=markers[year]['a'],
                s=s,
                linewidths=lw,
                label=f'{year} ({n:02})'  # f'{label}({n})'
            )

        # find over two and print filenames

        for case in ('over', 'under'):
            x, y = get_under_over(case=case
                                  , x=xvals
                                  , y=yvals
                                  , ratio=ratio[-1], )

            idx = xdata['x'].isin(x)
            if xdata.loc[idx, :].size > 0:
                file = f'{sen.replace("/", "_")}_{year}_{case}.csv'
                subset = xdata.loc[idx, :].reset_index()
                # print(ydata.loc[idx])
                subset['y'] = ydata.loc[idx].values
                subset.loc[:, ['Date', 'Time', 'Lat', 'Lon', 'station', 'x', 'y']].to_csv(file, index=False)
                print(file)
                print()

            c = 'r' if case == 'over' else 'b'
            f = c if markers[year]['f'] == 'k' else markers[year]['f']
            ax.scatter(
                x, y, s=s
                , marker=markers[year]['m']
                , facecolors=f
                , edgecolors=c
                , alpha=markers[year]['a']
                , linewidths=lw)

    ax.set_xlabel('$\t{In}$-$\it{situ}$ CHL [mg m$^{-3}$]')
    ax.set_ylabel(f'{sen} CHL [mg m$^{{-3}}$]')

    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_xticks(xtick)

    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.set_yticks(ytick)
    x0, x1 = xlim
    y0, y1 = ylim

    fmt = ticker.FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    # overestimated(ax)
    if over:
        x, y = overestimated(
            xdata=xdata
            , ydata=ydata
            , ratio=ratio[-1])
        ax.scatter(x, y, s=s, c='r', marker='o', alpha=0.5)

    # underestimated(ax)
    if under:
        x, y = underestimated(
            xdata=xdata
            , ydata=ydata
            , ratio=ratio[-1])
        ax.scatter(x, y, s=s, c='b', marker='o', alpha=0.5)

    for r in ratio:
        if r == 1:
            ax.plot([x0, x1]
                    , [y0, y1]
                    , '-k'
                    , linewidth=lw
                    , label='$x = y$')
        if r == 2:
            ax.plot([x0 * r, x1]
                    , [y0, y1 / r]
                    , ':k'
                    , linewidth=lw
                    , label='1:2/2:1')

            ax.plot([x0, x1 / r]
                    , [y0 * r, y1]
                    , ':k'
                    , linewidth=lw)
        if r == 3:
            ax.plot([x0 * r, x1]
                    , [y0, y1 / r]
                    , '-'
                    , color='#2F5597'
                    , linewidth=lw
                    , label='1:3/3:1')

            ax.plot([x0, x1 / r]
                    , [y0 * r, y1]
                    , '-'
                    , color='#2F5597'
                    , linewidth=lw)
    ax.legend()
    ax.grid(which='major', axis='both', linestyle=':')

    return ax


def merge_pd(left: pd, right: pd, append: bool = False):
    if left is None:
        return right
    if append:
        return left.append(right)
    return left.merge(right, how='left',
                      left_index=True,
                      right_index=True)


def get_plot(df
             , markers: dict
             , sen: str
             , var_select: str
             , w: int = 8
             , h: int = 5
             , lw: float = 1.
             ):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    xx, yy = [], []

    for mk in markers.keys():
        print(mk)

        dfi = df.loc[mk].dropna()
        x = np.asarray([float(v) for v in dfi.loc[:, 'chla'].to_numpy()])
        y = np.asarray([float(v) for v in dfi.loc[:, var_select].to_numpy()])

        racio = [1, 2, 3] if mk == 'China' else []
        ax = get_scatter(xdata=x
                         , ydata=y
                         , ax=ax
                         , ratio=racio
                         , color=markers[mk]['c']
                         , alpha=markers[mk]['a']
                         , fill=markers[mk]['f']
                         , marker=markers[mk]['m']
                         , label=mk
                         , lw=lw
                         , sen=sen)

        # print(np.log10(dfi.loc[:, 'chla'].to_numpy()))
        # print(x, y) if mk == 'China' else ''
        mask = np.isnan(x) | np.isnan(y)
        xx.extend(np.log10(x[~mask]).tolist())
        yy.extend(np.log10(y[~mask]).tolist())
    return np.asarray(xx), np.asarray(yy), ax


def merge_plot(path: Path
               , markers: dict
               , index: list
               , var_select: str
               , sen: str
               , field: str):
    # russia_dataset.2022-03-14T01-12-14-071984.GOCI.OC.matchup.csv
    for key, val in markers.items():
        # print(f'{key}')
        for file in path.glob(f'{key.lower()}*.GOCI.OC.matchup.csv'):
            print(f'\t{file}')
            if 'russia' not in file.name:
                continue

            var_field = 'CHLA [mg m^-3]' if key == 'SGLI' else 'chlor_a [mg m^-3]'
            print(f'\n{file.name}: {var_field}')

            df = pd.read_csv(file, skiprows=14)

            df.mask(df.isin([-999, '-999', 'n.d.']), inplace=True)

            idx = df[field].isin([var_field])
            subset = df.loc[idx, index + ['chla', var_select]]
            subset.set_index(index, inplace=True)

            temp = df.loc[idx, index + ['chla_0', var_select]]
            temp.set_index(index, inplace=True)
            temp.rename(columns={'chla_0': 'chla'}, inplace=True)
            print(f'Left: {subset.shape} | Right: {temp.shape}')

            subset = merge_pd(left=subset, right=temp, append=True)
            print(f'Joined: {subset.shape}')

            if subset.shape[0] == 0:
                continue

            # remove Wave Glider data (sta with WG)
            sta_values = subset.index.get_level_values('station')
            idx = np.asarray([False if ('WG' in sta) else True
                              for sta in sta_values])
            subset = subset.loc[idx, :]

            # w, h = 17, 14
            xx, yy, ax = get_plot(df=subset
                                  , w=14
                                  , h=12
                                  , lw=0.5
                                  , markers=markers
                                  , var_select=var_select
                                  , sen=sen)
            x, y, txt = correl(xi=xx, yi=yy)

            nl = ax.plot(x, y, '-r', label='linear regression', lw=3)
            ax.text(.015, 8.5, txt)
            plt.legend(loc='upper right', bbox_to_anchor=(1., 0.41), fontsize=25)
            #         if key in ('MODISA', 'VIIRSN'):
            #             print(key)
            #             ax.axes.set_xlabel('')
            #             ax.axes.set_xticklabels('')
            # plt.title('0 m', loc='left')
            # plt.tight_layout()
            fmt = ticker.FormatStrFormatter('%g')
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            plt.savefig(path.joinpath(file.name.replace('.csv', '_noWG.png')), dpi=200)
