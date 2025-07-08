import streamlit as st
import requests
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
from typing import List, Dict, Any, Optional

# --- 1. Конфигурация страницы и константы ---
st.set_page_config(
    layout="wide",
    page_title="ShAMLock: AML Dashboard",
    page_icon="🕵️"
)

API_BASE_URL = "http://api:8000"

# --- 2. Инициализация состояния сессии ---
def init_session_state():
    """Инициализирует переменные состояния сессии, если они не существуют."""
    placeholder = "Выберите аккаунт..."
    defaults = {
        'sender_filter': placeholder,
        'receiver_filter': placeholder,
        'selected_edge': None,
        'graph_data': None,
        'graph_depth': 1,
        'sender_color': '#00BFFF',
        'receiver_color': '#32CD32',
        'other_color': '#FFD700'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'sender_options' not in st.session_state:
        st.session_state.sender_options = [placeholder, "Все"] + get_accounts('sender')
    if 'receiver_options' not in st.session_state:
        st.session_state.receiver_options = [placeholder, "Все"] + get_accounts('receiver')

# --- 3. Слой для работы с API ---
@st.cache_data(ttl=300)
def api_request(endpoint: str, method: str = "GET", params: Dict = None, json_data: Dict = None) -> Optional[Any]:
    """Централизованная функция для выполнения запросов к API."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params) if method.upper() == "GET" else requests.post(url, json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to API at {url}.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} for {url}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

@st.cache_data(ttl=300)
def get_accounts(role: str) -> List[str]:
    """Получает список счетов по их роли (отправитель/получатель/все)."""
    accounts = api_request("/accounts/by_role", params={"role": role})
    return sorted(accounts) if accounts else []

@st.cache_data(ttl=300)
def get_transactions(sender: str, receiver: str) -> pd.DataFrame:
    """Получает транзакции и подготавливает их для отображения."""
    params = {}
    placeholder = "Выберите аккаунт..."
    if sender and sender not in ["Все", placeholder]:
        params['sender_name'] = sender
    if receiver and receiver not in ["Все", placeholder]:
        params['receiver_name'] = receiver
    
    data = api_request("/transactions/", params=params)
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    # Принудительно создаем колонки с именами счетов, предполагая, что 'sender' и 'receiver' всегда есть
    df['sender_account'] = df['sender'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    df['receiver_account'] = df['receiver'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    if 'sender' in df.columns and isinstance(df['sender'].iloc[0], dict):
        df['sender'] = df['sender'].apply(lambda x: x.get('name'))
    if 'receiver' in df.columns and isinstance(df['receiver'].iloc[0], dict):
        df['receiver'] = df['receiver'].apply(lambda x: x.get('name'))
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['Дата'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['Время'] = df['datetime'].dt.strftime('%H:%M:%S')
    if 'risk_score' in df.columns:
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
    return df

@st.cache_data(ttl=300)
def get_transaction_by_id(tx_id: int) -> Optional[Dict]:
    """Получает одну транзакцию по ее ID и предварительно обрабатывает ее."""
    data = api_request(f"/transactions/{tx_id}")
    if not data:
        return None
    timestamp_col_name = 'datetime' if 'datetime' in data else 'event_timestamp'
    if timestamp_col_name in data:
        try:
            dt = pd.to_datetime(data[timestamp_col_name])
            data['Дата'] = dt.strftime('%Y-%m-%d')
            data['Время'] = dt.strftime('%H:%M:%S')
        except (ValueError, TypeError):
            data['Дата'] = 'N/A'
            data['Время'] = 'N/A'
    if 'sender' in data and isinstance(data.get('sender'), dict):
        data['sender'] = data['sender'].get('name')
    if 'receiver' in data and isinstance(data.get('receiver'), dict):
        data['receiver'] = data['receiver'].get('name')
    if 'risk_score' in data:
        data['risk_score'] = pd.to_numeric(data['risk_score'], errors='coerce')
    return data

@st.cache_data(ttl=300)
def get_graph_for_accounts(account_ids: List[str], depth: int) -> Optional[Dict]:
    """Получает данные для графа из API."""
    return api_request("/graphs/", method="POST", json_data={"account_ids": account_ids, "depth": depth})

def clear_graph_cache():
    """Очищает кэш для функции получения данных графа."""
    get_graph_for_accounts.clear()

@st.cache_data(ttl=3600)
def get_max_depth(account_id: str) -> int:
    """Получает максимальную глубину графа для указанного счета из API."""
    if not account_id or account_id in ["Все", "Выберите аккаунт..."]:
        return 5
    
    max_depth = api_request(f"/graphs/max_depth/{account_id}")
    return int(max_depth) if max_depth is not None else 5



# --- 4. Компоненты интерфейса ---
def reset_filters():
    """Сбрасывает все фильтры и очищает соответствующие кеши для получения свежих данных."""
    get_transactions.clear()
    get_accounts.clear()
    get_graph_for_accounts.clear()
    get_transaction_by_id.clear()

    placeholder = "Выберите аккаунт..."
    st.session_state.sender_filter = placeholder
    st.session_state.receiver_filter = placeholder
    st.session_state.sender_options = [placeholder, "Все"] + get_accounts('sender')
    st.session_state.receiver_options = [placeholder, "Все"] + get_accounts('receiver')
    st.session_state.selected_edge = None
    st.session_state.graph_data = None

def update_receiver_options():
    """Вызывается при изменении фильтра отправителя. Обновляет опции для получателя."""
    placeholder = "Выберите аккаунт..."
    sender = st.session_state.sender_filter
    
    if sender and sender not in [placeholder, "Все"]:
        df = get_transactions(sender=sender, receiver="Все")
        new_options = [placeholder, "Все"]
        if not df.empty:
            valid_receivers = sorted(pd.unique(df['receiver'].dropna()).tolist())
            new_options.extend(valid_receivers)
        
        st.session_state.receiver_options = new_options
        if st.session_state.receiver_filter not in new_options:
            st.session_state.receiver_filter = placeholder
    else:
        st.session_state.receiver_options = [placeholder, "Все"] + get_accounts('receiver')

def update_sender_options():
    """Вызывается при изменении фильтра получателя. Обновляет опции для отправителя."""
    placeholder = "Выберите аккаунт..."
    receiver = st.session_state.receiver_filter

    if receiver and receiver not in [placeholder, "Все"]:
        df = get_transactions(sender="Все", receiver=receiver)
        new_options = [placeholder, "Все"]
        if not df.empty:
            valid_senders = sorted(pd.unique(df['sender'].dropna()).tolist())
            new_options.extend(valid_senders)

        st.session_state.sender_options = new_options
        if st.session_state.sender_filter not in new_options:
            st.session_state.sender_filter = placeholder
    else:
        st.session_state.sender_options = [placeholder, "Все"] + get_accounts('sender')

def display_filters():
    """Отображает взаимозависимые фильтры, обеспечивая корректное состояние интерфейса."""
    st.sidebar.header("Фильтры")

    sender_options = st.session_state.get('sender_options', [])
    sender_value = st.session_state.get('sender_filter')
    try:
        sender_index = sender_options.index(sender_value)
    except ValueError:
        sender_index = 0

    st.sidebar.selectbox(
        "Отправитель:",
        options=sender_options,
        index=sender_index,
        key='sender_filter', 
        on_change=update_receiver_options
    )

    receiver_options = st.session_state.get('receiver_options', [])
    receiver_value = st.session_state.get('receiver_filter')
    try:
        receiver_index = receiver_options.index(receiver_value)
    except ValueError:
        receiver_index = 0

    st.sidebar.selectbox(
        "Получатель:",
        options=receiver_options,
        index=receiver_index,
        key='receiver_filter',
        on_change=update_sender_options
    )

    st.sidebar.button("Сбросить фильтры", on_click=reset_filters)

@st.cache_data(ttl=300)
def get_max_depth_from_api(account_id: str) -> int:
    """Получает максимальную глубину для счета из API."""
    if not account_id:
        return 1
    response = api_request(f"/graphs/max_depth/{account_id}", method="GET")
    # API возвращает просто число, а не JSON-объект
    return int(response) if response is not None else 1

def display_graph_tab(df: pd.DataFrame):
    st.subheader("Анализ графа транзакций")

    # Используем глобальные фильтры из сайдбара
    sender = st.session_state.get('sender_filter')
    receiver = st.session_state.get('receiver_filter')
    
    # Определяем плейсхолдеры
    placeholder = "Выберите аккаунт..."
    all_filter = "Все"

    # Определяем, какой аккаунт выбран для анализа
    account_to_analyze = None
    if sender and sender not in [placeholder, all_filter]:
        account_to_analyze = sender
    elif receiver and receiver not in [placeholder, all_filter]:
        account_to_analyze = receiver

    # Если ни один конкретный аккаунт не выбран, показываем инструкцию
    if not account_to_analyze:
        st.info("Выберите отправителя или получателя в боковой панели, чтобы построить граф транзакций.")
        return

    # --- Основной макет: 2 колонки ---
    graph_col, settings_col = st.columns([3, 1])

    with settings_col:
        st.markdown("#### Настройки графа")

        # --- Динамический слайдер глубины ---
        max_depth = get_max_depth_from_api(account_to_analyze)
        if max_depth > 1:
            depth = st.slider("**Глубина анализа**", min_value=1, max_value=max_depth, value=min(1, max_depth), key="depth_slider")
        else:
            st.markdown("**Глубина графа: 1**")
            depth = 1

        # --- Настройки цветов ---
        st.markdown("**Цвета узлов**", help="Выбор цветов узлов на графике")
        st.session_state.sender_color = st.color_picker("Отправитель", st.session_state.get('sender_color', '#FF6347'))
        st.session_state.receiver_color = st.color_picker("Получатель", st.session_state.get('receiver_color', '#4682B4'))
        st.session_state.other_color = st.color_picker("Другие", st.session_state.get('other_color', '#D3D3D3'))
        
        # --- Легенда ---
        st.markdown("**Легенда**")
        legend_html = f'''
        <ul style='list-style-type: none; padding-left: 0; font-size: 14px;'>
            <li style='display: flex; align-items: center; margin-bottom: 5px;'>
                <div style='width: 15px; height: 3px; background-color: maroon; margin-right: 10px;'></div>
                <span>Алертная транзакция</span>
            </li>
            <li style='display: flex; align-items: center;'>
                <div style='width: 15px; height: 3px; background-color: gray; margin-right: 10px;'></div>
                <span>Обычная транзакция</span>
            </li>
        </ul>
        '''
        st.markdown(legend_html, unsafe_allow_html=True)

    with graph_col:
        with st.container(border=True):
            with st.spinner("Построение графа..."):
                graph_data = get_graph_for_accounts(account_ids=[account_to_analyze], depth=depth)

                if graph_data and graph_data.get('nodes'):
                    def get_node_color(name):
                        if name == sender: return st.session_state.sender_color
                        if name == receiver: return st.session_state.receiver_color
                        return st.session_state.other_color

                    nodes = [Node(id=str(n['id']), label=' ', color=get_node_color(n['name']), size=25) for n in graph_data['nodes']]

                    from collections import defaultdict
                    edge_groups = defaultdict(list)
                    for e in graph_data.get('edges', []):
                        edge_groups[(e['source'], e['target'])].append(e)

                    edges = []
                    for (source, target), transactions in edge_groups.items():
                        # Для основной информации используем первую транзакцию
                        first_tx = transactions[0]

                        tooltip = (
                            f"Отправитель: {first_tx['source']}\n"
                            f"Получатель: {first_tx['target']}\n"
                            f"Дата: {first_tx.get('date', 'N/A')}\n"
                            f"Время: {first_tx.get('time', 'N/A')}\n"
                            f"Сумма: {first_tx.get('amount', 0):,.2f} {first_tx.get('currency', '')}".replace(',', ' ')
                        )

                        # Если транзакций между узлами больше одной, добавляем подпись
                        if len(transactions) > 1:
                            tooltip += f"\n\nВсего транзакций: {len(transactions)}. \nОстальные см. во вкладке 'Список транзакций'"

                        # Если хотя бы одна транзакция мошенническая, ребро будет красным
                        is_fraud = any(tx.get('is_fraud') for tx in transactions)

                        edges.append(Edge(
                            source=source,
                            target=target,
                            color='maroon' if is_fraud else 'gray',
                            title=tooltip
                        ))

                    config = Config(
                        width=850,
                        height=600,
                        directed=True, # Оставляем для корректного лэйаута
                        physics=True,
                        nodeHighlightBehavior=True,
                        highlightColor="#F7A7A6",
                        collapsible=True,
                        node={'labelProperty':'label'},
                        # Настройки для ребер (стрелок)
                        edges={
                            "arrows": {
                                "to": {"enabled": True, "scaleFactor": 0.7},
                                "from": {"enabled": True, "scaleFactor": 0.7} # Добавляем стрелку 'откуда'
                            }
                        }
                    )
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.info("Для построения графа нет данных.")

def display_transactions_tab(df: pd.DataFrame):
    """Форматирует и отображает основную таблицу транзакций в соответствии с заданными спецификациями."""
    if df.empty:
        st.info("Выберите фильтры, чтобы увидеть транзакции.")
        return

    df_display = df.copy()

    timestamp_col = next((col for col in ['datetime', 'event_timestamp'] if col in df_display.columns), None)
    if timestamp_col:
        try:
            df_display[timestamp_col] = pd.to_datetime(df_display[timestamp_col])
            df_display['Дата'] = df_display[timestamp_col].dt.date
            df_display['Время'] = df_display[timestamp_col].dt.strftime('%H:%M:%S')
        except Exception as e:
            st.warning(f"Не удалось обработать колонку времени: {e}")
            df_display['Дата'] = 'Error'
            df_display['Время'] = 'Error'

    cols_to_drop = ['id', 'from_account_id', 'to_account_id', 'fraud_probability', timestamp_col]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_display.columns]
    df_display.drop(columns=existing_cols_to_drop, inplace=True)

    rename_map = {
        'sender': 'Аккаунт отправителя',
        'receiver': 'Аккаунт получателя',
        'amount': 'Размер транзакции',
        'transaction_type': 'Тип',
        'payment_currency_iso': 'Валюта отправления',
        'received_currency_iso': 'Валюта получения',
        'payment_type': 'Тип операции',
        'sender_bank_location': 'Страна отправления',
        'receiver_bank_location': 'Страна получения',
        'is_fraud': 'Отмывка',
        'risk_score': 'Вероятность мошенничества'
    }
    df_display.rename(columns=rename_map, inplace=True)

    start_cols = [
        'Аккаунт отправителя',
        'Аккаунт получателя',
        'Дата',
        'Время',
        'Размер транзакции',
        'Тип'
    ]
    end_col = 'Отмывка'

    final_cols = [col for col in start_cols if col in df_display.columns]
    other_cols = [col for col in df_display.columns if col not in final_cols and col != end_col]
    final_cols.extend(other_cols)
    if end_col in df_display.columns:
        final_cols.append(end_col)

    df_to_show = df_display[final_cols]

    format_dict = {}
    if 'Вероятность мошенничества' in df_to_show.columns:
        if pd.api.types.is_numeric_dtype(df_to_show['Вероятность мошенничества']):
            format_dict['Вероятность мошенничества'] = '{:.2%}'
    if 'Размер транзакции' in df_to_show.columns:
        format_dict['Размер транзакции'] = '{:,.2f}'

    st.dataframe(df_to_show.style.format(format_dict), use_container_width=True)

def display_alerts_tab(df: pd.DataFrame):
    """Отображает таблицу транзакций с градиентной расцветкой по риску."""
    st.subheader("Транзакции с оценкой риска")

    if df.empty:
        st.info("Выберите фильтры, чтобы увидеть транзакции.")
        return

    display_df = df.copy()

    # --- Подготовка данных для отображения ---
    # 1. Дата и время
    timestamp_col = next((col for col in ['datetime', 'event_timestamp'] if col in display_df.columns), None)
    if timestamp_col:
        try:
            dt_series = pd.to_datetime(display_df[timestamp_col])
            display_df['Дата'] = dt_series.dt.date
            display_df['Время'] = dt_series.dt.strftime('%H:%M:%S')
        except Exception:
            display_df['Дата'] = 'N/A'
            display_df['Время'] = 'N/A'

    # 2. Вероятность мошенничества (оставляем числом для стилизации)
    prob_col = 'fraud_probability' if 'fraud_probability' in display_df.columns else 'risk_score'
    if prob_col in display_df.columns:
        display_df['Вероятность мошенничества'] = pd.to_numeric(display_df[prob_col], errors='coerce')
    else:
        display_df['Вероятность мошенничества'] = float('nan')

    # 3. Переименование колонок
    display_df.rename(columns={'sender': 'Отправитель', 'receiver': 'Получатель'}, inplace=True)

    # --- Отображение таблицы со стилем ---
    final_cols_order = ['Отправитель', 'Получатель', 'Дата', 'Время', 'Вероятность мошенничества']
    cols_to_display = [col for col in final_cols_order if col in display_df.columns]
    
    df_to_show = display_df[cols_to_display]

    st.dataframe(
        df_to_show.style.background_gradient(
            cmap='RdYlGn_r',
            subset=['Вероятность мошенничества'],
            vmin=0.0,
            vmax=1.0
        ).format(
            {'Вероятность мошенничества': '{:.2%}'},
            na_rep="-"
        ),
        use_container_width=True,
        hide_index=True
    )

# --- 5. Основной поток приложения ---
def main():
    """Основная функция для запуска приложения Streamlit."""
    st.title("🕵️ ShAMLock: AML Dashboard")
    st.markdown("Панель для мониторинга и анализа финансовых транзакций.")
    st.markdown('''
        <style>
            .stColorPicker > div {
                width: 1.5rem !important;
                height: 1.5rem !important;
            }
            .stColorPicker > div > div {
                width: 1.5rem !important;
                height: 1.5rem !important;
            }
        </style>
    ''', unsafe_allow_html=True)

    init_session_state()
    display_filters()
    transactions_df = get_transactions(st.session_state.sender_filter, st.session_state.receiver_filter)
    
    # CSS для стилизации st.radio под вкладки
    st.markdown("""
    <style>
        div[role="radiogroup"] {
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f0f2f6;
        }
        div[role="radiogroup"] > label {
            display: inline-block;
            padding: 8px 16px;
            margin: 0;
            background-color: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
        }
        div[role="radiogroup"] > label > div:first-child {
            display: none; /* Скрываем точку радио-кнопки */
        }
        div[role="radiogroup"] > label > div:last-child {
            font-weight: 500;
        }
        /* Стиль для активной "вкладки" */
        div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
            border-bottom: 2px solid #FF4B4B; /* Цвет активной вкладки */
            color: #FF4B4B;
        }
    </style>
    """, unsafe_allow_html=True)

    tab_names = ["Список транзакций", "Анализ графа", "Алерты"]

    # Используем st.radio для навигации, чтобы сохранять состояние вкладки
    active_tab = st.radio(
        "Навигация", 
        tab_names, 
        key='active_tab', 
        horizontal=True,
        label_visibility="collapsed"
    )

    # Отображаем содержимое в зависимости от выбранной "вкладки"
    if active_tab == "Список транзакций":
        display_transactions_tab(transactions_df)
    elif active_tab == "Анализ графа":
        display_graph_tab(transactions_df)
    elif active_tab == "Алерты":
        display_alerts_tab(transactions_df)

if __name__ == "__main__":
    main()
