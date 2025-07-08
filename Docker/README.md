# Docker Configuration

Этот каталог содержит `Dockerfile` для сборки Docker-образов различных сервисов проекта **ShAMLock**.

Эти файлы используются `docker-compose.yml` в корневом каталоге для оркестрации и запуска всего приложения.

## Файлы

-   `api.Dockerfile`: Dockerfile для сборки образа **FastAPI бэкенда**.
    -   Устанавливает Python и необходимые зависимости из `requirements.txt`.
    -   Копирует исходный код из папки `app`.
    -   Запускает веб-сервер `uvicorn` для обслуживания API.

-   `dashboard.Dockerfile`: Dockerfile для сборки образа **Streamlit дашборда**.
    -   Устанавливает Python и зависимости.
    -   Копирует файл `dashboard.py`.
    -   Запускает дашборд с помощью команды `streamlit run`.
