import networkx as nx
import matplotlib.pyplot as plt
import math

def visualize_laundering_types_multigraph_grid(MG, max_nodes=40, columns=2):
    laundering_types = sorted(set(
        d['Laundering_type'] for _, _, _, d in MG.edges(keys=True, data=True)
        if 'Laundering_type' in d
    ))

    num_types = len(laundering_types)
    rows = math.ceil(num_types / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows))
    axes = axes.flatten()

    for idx, ltype in enumerate(laundering_types):
        ax = axes[idx]

        # Отбор рёбер с этим типом
        edges = [(u, v, k) for u, v, k, d in MG.edges(keys=True, data=True)
                 if d.get('Laundering_type') == ltype]

        if not edges:
            ax.set_title(f"{ltype}\n(пусто)")
            ax.axis('off')
            continue

        # Подграф
        subG = nx.MultiDiGraph()
        for u, v, k in edges:
            subG.add_edge(u, v, **MG[u][v][k])

        # Выбираем компоненту ≤ max_nodes
        components = [subG.subgraph(c).copy() for c in nx.weakly_connected_components(subG.to_directed())]
        components = sorted(components, key=lambda g: g.number_of_nodes(), reverse=True)

        for comp in components:
            if comp.number_of_nodes() <= max_nodes:
                selected = comp
                break
        else:
            ax.set_title(f"{ltype}\n(слишком большой)")
            ax.axis('off')
            continue

        # Определяем is_laundering статус
        laundering_flags = {d.get('Is_laundering') for _, _, _, d in selected.edges(keys=True, data=True)}
        if laundering_flags == {1}:
            laundering_flag = "Is_laundering = 1"
        elif laundering_flags == {0}:
            laundering_flag = "Is_laundering = 0"
        else:
            laundering_flag = "Is_laundering = Mixed"

        # Расчёт layout
        pos = nx.spring_layout(selected, seed=42)

        # Цвет рёбер
        edge_colors = [
            'red' if d.get('Is_laundering') == 1 else 'gray'
            for _, _, _, d in selected.edges(keys=True, data=True)
        ]

        # Визуализация узлов и рёбер
        nx.draw_networkx_nodes(selected, pos, node_color='lightgreen', node_size=500, ax=ax)
        nx.draw_networkx_edges(
            selected,
            pos,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle='->',
            arrowsize=15,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

        # Обезличенные подписи узлов
        node_id_map = {node: str(i+1) for i, node in enumerate(selected.nodes())}
        nx.draw_networkx_labels(selected, pos, labels=node_id_map, font_size=8, ax=ax)

        # Горизонтальные подписи рёбер (дата + округлённая сумма)
        for u, v, k, d in selected.edges(keys=True, data=True):
            date = d.get('Date', '')
            amount = round(d.get('Amount', 0))
            label = f"{date}\n{amount} GBP"
            color = 'red' if d.get('Is_laundering') == 1 else 'black'
            x, y = (pos[u] + pos[v]) / 2
            ax.text(x, y + 0.03, label, fontsize=6, color=color, ha='center', va='bottom', rotation=0)

        # Заголовок
        ax.set_title(f"{ltype}\n({laundering_flag})", fontsize=10)
        ax.axis('off')

    # Пустые ячейки
    for i in range(num_types, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
