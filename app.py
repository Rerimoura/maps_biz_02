# -*- coding: utf-8 -*-
# App Streamlit: subir clientes + endereços, visualizar no mapa, traçar rota e exportar GeoJSON
# Autor: Você :)
# Observação: comentários e rótulos em PT-BR para facilitar manutenção

import math
import json
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List

# Geocoders (geopy) – suportam Nominatim (gratuito), Google e OpenCage
from geopy.geocoders import Nominatim, GoogleV3, OpenCage
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Visualização em mapa
import pydeck as pdk

# Rotas (OSRM)
import requests

# ==============================
# Configuração da Página
# ==============================
st.set_page_config(page_title="Clientes no Mapa + Rotas + GeoJSON", layout="wide")

st.title("📍 Clientes no Mapa + 🛣️ Rotas + 🌐 GeoJSON")
st.caption(
    "Envie uma planilha (CSV/XLSX) com **Cliente** e **Endereço**, geocodifique, visualize no mapa, "
    "trace uma rota (linhas retas ou OSRM) e exporte **GeoJSON**."
)

# ==============================
# Estado (session_state) helpers
# ==============================
def _init_state():
    if "df_geo" not in st.session_state:
        st.session_state.df_geo = None
    if "route" not in st.session_state:
        st.session_state.route = None  # dict com: coords_lonlat, ordem_df, distancia_km, duracao_min, perfil, fechar

def reset_geocoded():
    st.session_state.df_geo = None
    st.session_state.route = None

def reset_route():
    st.session_state.route = None

_init_state()

# ==============================
# Funções utilitárias
# ==============================
def ler_arquivo(uploaded_file) -> pd.DataFrame:
    """
    Lê CSV/XLSX e tenta ser tolerante a separadores e encoding.
    Remove colunas duplicadas mantendo a primeira ocorrência.
    """
    nome = uploaded_file.name.lower()
    if nome.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    elif nome.endswith(".xlsx") or nome.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato não suportado. Envie .csv, .xlsx ou .xls")

    # Remove colunas duplicadas, se houver
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def inicializar_geocoder(
    provedor: str,
    api_key: Optional[str],
    country_bias: Optional[str],
    language: str = "pt-BR",
    user_agent_email: Optional[str] = None,
):
    """
    Retorna um geocoder do geopy conforme provedor selecionado.
    """
    provedor_low = provedor.lower()
    if provedor_low == "nominatim (gratuito)":
        agent = f"clientes-mapa-app/1.0 ({user_agent_email or 'seu_email@exemplo.com'})"
        geocoder = Nominatim(user_agent=agent, timeout=10)
        return geocoder, {"language": language, "country_codes": (country_bias or "").lower()}
    elif provedor_low == "google":
        if not api_key:
            st.error("Informe a chave de API do Google para usar este provedor.")
            st.stop()
        geocoder = GoogleV3(api_key=api_key, timeout=10)
        return geocoder, {"language": language, "region": country_bias}
    elif provedor_low == "opencage":
        if not api_key:
            st.error("Informe a chave de API do OpenCage para usar este provedor.")
            st.stop()
        geocoder = OpenCage(api_key=api_key, timeout=10)
        return geocoder, {"language": language, "countrycode": (country_bias or "").lower()}
    else:
        st.error("Provedor inválido.")
        st.stop()


@st.cache_data(show_spinner=False)
def geocodificar_endereco_cache(
    endereco: str,
    provedor: str,
    api_key: Optional[str],
    country_bias: Optional[str],
    language: str = "pt-BR",
    user_agent_email: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Geocodifica 1 endereço e retorna (lat, lon, status).
    Usa cache do Streamlit para não repetir a mesma consulta.
    Status: 'OK' ou mensagem de erro.
    """
    geocoder, params = inicializar_geocoder(
        provedor=provedor,
        api_key=api_key,
        country_bias=country_bias,
        language=language,
        user_agent_email=user_agent_email,
    )
    try:
        min_delay = 1.0 if provedor.lower().startswith("nominatim") else 0.0
        geocode = RateLimiter(geocoder.geocode, min_delay_seconds=min_delay, swallow_exceptions=True)
        loc = geocode(endereco, **{k: v for k, v in params.items() if v})
        if loc:
            return loc.latitude, loc.longitude, "OK"
        return None, None, "Não encontrado"
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None, None, "Serviço indisponível/timeout"
    except Exception as e:
        return None, None, f"Erro: {e}"


def geocodificar_dataframe(
    df: pd.DataFrame,
    col_cliente: str,
    col_endereco: str,
    provedor: str,
    api_key: Optional[str],
    country_bias: Optional[str],
    language: str = "pt-BR",
    user_agent_email: Optional[str] = None,
) -> pd.DataFrame:
    """
    Geocodifica todas as linhas do DataFrame respeitando cache e rate limit.
    Remove duplicados de endereço para ganhar tempo e depois reatribui.
    Implementação robusta que evita colunas duplicadas no sub-DataFrame.
    """
    if col_cliente not in df.columns or col_endereco not in df.columns:
        raise KeyError("Coluna selecionada não encontrada no DataFrame após a leitura.")

    work = pd.DataFrame({
        "cliente": df[col_cliente].astype("string").fillna(""),
        "endereco": df[col_endereco].astype("string").fillna(""),
    })

    # Normalização para o cache
    work["endereco_norm"] = work["endereco"].str.strip().str.lower()

    # Remove endereços vazios antecipadamente
    work["endereco_vazio"] = work["endereco"].str.strip().eq("")
    if work["endereco_vazio"].all():
        st.warning("Todas as linhas estão com endereço vazio. Preencha a coluna de endereços.")
        return work.assign(lat=pd.NA, lon=pd.NA, status="Endereço vazio").drop(columns=["endereco_vazio"])

    # Endereços únicos (não vazios)
    unicos = work.loc[~work["endereco_vazio"]].drop_duplicates(subset=["endereco_norm"]).copy()

    resultados = []
    progress = st.progress(0, text="Geocodificando endereços...")
    total = len(unicos)
    for i, row in enumerate(unicos.itertuples(index=False)):
        endereco = row.endereco
        lat, lon, status = geocodificar_endereco_cache(
            endereco=endereco,
            provedor=provedor,
            api_key=api_key,
            country_bias=country_bias,
            language=language,
            user_agent_email=user_agent_email,
        )
        resultados.append((row.endereco_norm, lat, lon, status))
        if total > 0:
            progress.progress(int((i + 1) / total * 100), text=f"Geocodificando ({i+1}/{total})")

    progress.empty()
    res_df = pd.DataFrame(resultados, columns=["endereco_norm", "lat", "lon", "status"])

    # Junta de volta no DataFrame original
    final = work.merge(res_df, on="endereco_norm", how="left").drop(columns=["endereco_norm", "endereco_vazio"])
    return final


# ==============================
# Cálculo de rotas (sem API) – heurísticas
# ==============================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distância Haversine em quilômetros."""
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def rota_distancia_km(coords: List[Tuple[float, float]], ordem: List[int], fechar: bool) -> float:
    """Distância total (km) para a ordem fornecida em coords=[(lat,lon), ...]."""
    if len(ordem) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(ordem) - 1):
        a, b = ordem[i], ordem[i+1]
        dist += haversine_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
    if fechar and len(ordem) > 2:
        dist += haversine_km(coords[ordem[-1]][0], coords[ordem[-1]][1], coords[ordem[0]][0], coords[ordem[0]][1])
    return dist


def nearest_neighbor_order(coords: List[Tuple[float, float]], start_index: int = 0) -> List[int]:
    """Ordem inicial por Vizinho Mais Próximo (NN)."""
    n = len(coords)
    if n == 0:
        return []
    nao_visitados = set(range(n))
    ordem = [start_index]
    nao_visitados.remove(start_index)
    atual = start_index
    while nao_visitados:
        prox = min(nao_visitados, key=lambda j: haversine_km(coords[atual][0], coords[atual][1], coords[j][0], coords[j][1]))
        ordem.append(prox)
        nao_visitados.remove(prox)
        atual = prox
    return ordem


def two_opt(coords: List[Tuple[float, float]], ordem: List[int], max_iter: int = 200) -> List[int]:
    """Melhora a ordem (rota aberta) usando 2-opt básico."""
    if len(ordem) < 4:
        return ordem[:]
    best = ordem[:]
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                d_before = (
                    haversine_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1]) +
                    haversine_km(coords[c][0], coords[c][1], coords[d][0], coords[d][1])
                )
                d_after = (
                    haversine_km(coords[a][0], coords[a][1], coords[c][0], coords[c][1]) +
                    haversine_km(coords[b][0], coords[b][1], coords[d][0], coords[d][1])
                )
                if d_after + 1e-9 < d_before:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
    return best


def construir_ordem(coords: List[Tuple[float, float]], metodo: str) -> List[int]:
    """Constroi ordem conforme método."""
    if not coords:
        return []
    if metodo == "Pela ordem do arquivo":
        return list(range(len(coords)))
    elif metodo == "Otimizar (Vizinho + 2-opt)":
        nn = nearest_neighbor_order(coords, start_index=0)
        return two_opt(coords, nn, max_iter=200)
    else:
        return list(range(len(coords)))


# ==============================
# Rotas com OSRM (API pública)
# ==============================
def osrm_route(coords: List[Tuple[float, float]], profile: str = "driving"):
    """
    Pede rota ao OSRM público.
    coords: lista [(lat, lon), ...]
    Retorna: (geometry_coords_lonlat, distancia_m, duracao_s)
    """
    if len(coords) < 2:
        raise ValueError("São necessários pelo menos 2 pontos para rota.")
    coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for (lat, lon) in coords])  # OSRM espera lon,lat
    url = f"https://router.project-osrm.org/route/v1/{profile}/{coords_str}?overview=full&geometries=geojson&steps=false"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        raise ValueError(data.get("message", "Erro ao solicitar rota no OSRM."))
    route = data["routes"][0]
    geometry = route["geometry"]["coordinates"]  # [[lon, lat], ...]
    distancia_m = route.get("distance", None)
    duracao_s = route.get("duration", None)
    return geometry, distancia_m, duracao_s


# ==============================
# GeoJSON: LineString (rota) + Points (waypoints)
# ==============================
def montar_geojson_rota(
    route_coords_lonlat,
    pontos_ordem_df: pd.DataFrame,
    distancia_km: Optional[float] = None,
    duracao_min: Optional[float] = None,
    profile: Optional[str] = None,
    fechar_rota: bool = False,
) -> str:
    """
    Monta um FeatureCollection GeoJSON com:
      - 1 LineString: a rota (coordinates em [lon, lat])
      - N Points: waypoints na ordem da rota
    """
    features = []

    if route_coords_lonlat and len(route_coords_lonlat) >= 2:
        features.append({
            "type": "Feature",
            "properties": {
                "tipo": "rota",
                "distancia_km": round(float(distancia_km), 3) if distancia_km is not None else None,
                "duracao_min": round(float(duracao_min), 1) if duracao_min is not None else None,
                "profile": profile,
                "fechar_rota": fechar_rota,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": route_coords_lonlat  # [[lon, lat], ...]
            }
        })

    for ordem_idx, row in pontos_ordem_df.reset_index(drop=True).iterrows():
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except Exception:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "tipo": "ponto",
                "ordem": int(ordem_idx) + 1,
                "cliente": str(row.get("cliente", "")),
                "endereco": str(row.get("endereco", "")),
                "status": str(row.get("status", "")),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })

    fc = {"type": "FeatureCollection", "features": features}
    return json.dumps(fc, ensure_ascii=False, indent=2)


def mostrar_mapa(df_geo: pd.DataFrame, route_coords_lonlat: Optional[List[List[float]]] = None):
    """Renderiza os pontos e, opcionalmente, a rota no mapa usando pydeck."""
    pontos = df_geo.dropna(subset=["lat", "lon"]).copy()
    if pontos.empty:
        st.warning("Nenhum ponto válido para plotar no mapa.")
        return

    mean_lat = float(pontos["lat"].astype(float).mean())
    mean_lon = float(pontos["lon"].astype(float).mean())

    layers = []

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=pontos,
        get_position="[lon, lat]",
        get_fill_color=[0, 122, 255, 200],
        get_radius=80,
        pickable=True,
    )
    layers.append(layer_points)

    if route_coords_lonlat and len(route_coords_lonlat) >= 2:
        path_data = [{"path": route_coords_lonlat, "name": "Rota"}]
        layer_route = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color=[255, 0, 0],
            width_scale=1,
            width_min_pixels=3,
            pickable=False,
        )
        layers.append(layer_route)

    tooltip = {
        "html": "<b>Cliente:</b> {cliente}<br/><b>Endereço:</b> {endereco}<br/><b>Status:</b> {status}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=10)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(deck)


# ==============================
# Barra lateral (opções)
# ==============================
with st.sidebar:
    st.header("⚙️ Configurações")

    st.markdown("**Provedor de geocodificação**")
    provedor = st.selectbox(
        "Escolha o provedor:",
        options=["Nominatim (gratuito)", "Google", "OpenCage"],
        index=0,
        help="Nominatim é grátis, mas tem limites (≈1 req/s). Google/OpenCage exigem chave de API.",
    )

    api_key = None
    if provedor in ("Google", "OpenCage"):
        api_key = st.text_input("Chave de API", type="password")

    country_bias = st.text_input(
        "País (bias opcional, ex.: BR, US, PT)", value="BR", help="Ajuda a priorizar resultados no país."
    )
    language = st.selectbox("Idioma de retorno", options=["pt-BR", "pt-PT", "en"], index=0)

    user_agent_email = None
    if provedor.startswith("Nominatim"):
        user_agent_email = st.text_input(
            "E-mail para user_agent (recomendado pelo Nominatim)", value="", help="Use um e-mail real."
        )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Limpar cache de geocodificação"):
            st.cache_data.clear()
            st.success("Cache limpo!")
    with col_b:
        if st.button("♻️ Limpar resultados (geocodificação/rota)"):
            reset_geocoded()
            st.success("Resultados limpos!")


st.divider()

# ==============================
# Upload do arquivo
# ==============================
exemplo = st.expander("Ver exemplo de formato de entrada (CSV)", expanded=False)
with exemplo:
    st.code(
        "Cliente,Endereco\n"
        "Cliente A,Av. Paulista 1578, São Paulo - SP\n"
        "Cliente B,Rua XV de Novembro 50, Curitiba - PR\n"
        "Cliente C,Esplanada dos Ministérios, Brasília - DF\n",
        language="csv",
    )

arquivo = st.file_uploader("Envie seu arquivo (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

if arquivo:
    try:
        df = ler_arquivo(arquivo)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    st.success(f"Arquivo carregado com {len(df):,} linhas e {len(df.columns)} colunas.")
    st.dataframe(df.head(10), use_container_width=True)

    # Selecionar colunas
    colunas = list(df.columns)
    if not colunas:
        st.error("Nenhuma coluna encontrada no arquivo.")
        st.stop()

    col_cliente = st.selectbox("Coluna do **Cliente**", options=colunas, key="col_cliente")
    col_endereco = st.selectbox("Coluna do **Endereço**", options=colunas, key="col_endereco")

    # Validação: impedir escolher a mesma coluna
    if col_cliente == col_endereco:
        st.error("Selecione **colunas diferentes** para Cliente e Endereço.")
        st.stop()

    # -------------------------------
    # Botão: Geocodificar (salva no state)
    # -------------------------------
    if st.button("Geocodificar e Mostrar no Mapa", type="primary", key="btn_geocodificar"):
        with st.spinner("Processando endereços..."):
            df_geo = geocodificar_dataframe(
                df=df,
                col_cliente=col_cliente,
                col_endereco=col_endereco,
                provedor=provedor,
                api_key=api_key,
                country_bias=country_bias,
                language=language,
                user_agent_email=user_agent_email or None,
            )
        st.session_state.df_geo = df_geo  # <- PERSISTE RESULTADO
        st.session_state.route = None     # limpa rota anterior
        st.success("Geocodificação concluída! Role a página para a seção de rotas.")

# ==============================
# Se existe geocodificação salva, mostra resultados e permite rotas
# (Fora do if do botão, para persistir após rerun)
# ==============================
if st.session_state.df_geo is not None:
    df_geo = st.session_state.df_geo

    # Métricas de qualidade
    total = len(df_geo)
    ok = (df_geo["status"] == "OK").sum()
    not_found = (df_geo["status"] == "Não encontrado").sum()
    vazios = (df_geo["status"] == "Endereço vazio").sum()
    erros = total - ok - not_found - vazios

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de registros", f"{total:,}")
    m2.metric("Geocodificados", f"{ok:,}")
    m3.metric("Não encontrados", f"{not_found:,}")
    m4.metric("Vazios/Erros", f"{(vazios + max(erros, 0)):,}")

    # ==============================
    # Seção: Rotas (sempre visível quando df_geo existe)
    # ==============================
    st.subheader("🛣️ Rota entre clientes (opcional)")

    pontos_validos = df_geo.dropna(subset=["lat", "lon"]).reset_index(drop=True).copy()
    if len(pontos_validos) < 2:
        st.info("São necessários pelo menos **2** pontos geocodificados para traçar uma rota.")
        mostrar_mapa(df_geo)
    else:
        # Controles de rota (fora de forms para simplificar; usamos session_state para resultados)
        col1, col2, col3 = st.columns(3)
        with col1:
            metodo_ordem = st.selectbox(
                "Método de ordenação",
                options=["Pela ordem do arquivo", "Otimizar (Vizinho + 2-opt)"],
                help="Escolha como a ordem dos clientes será definida.",
                key="opt_metodo_ordem"
            )
        with col2:
            fechar_rota = st.checkbox("Voltar ao início (rota fechada)", value=False, key="opt_fechar")
        with col3:
            motor_rota = st.selectbox(
                "Motor de rota",
                options=["Linhas retas (sem API)", "OSRM público (gratuito)"],
                help="OSRM retorna distância e tempo reais; Linhas retas estimam distância geodésica.",
                key="opt_motor"
            )

        perfil_osrm = None
        if motor_rota == "OSRM público (gratuito)":
            perfil_osrm = st.selectbox("Perfil de rota (OSRM)", options=["driving", "walking", "cycling"], index=0, key="opt_perfil")

        colb1, colb2 = st.columns(2)
        with colb1:
            gerar = st.button("Gerar rota", type="secondary", key="btn_gerar_rota")
        with colb2:
            if st.button("Limpar rota", key="btn_limpar_rota"):
                reset_route()
                st.info("Rota limpa.")

        # Se clicar em gerar, calcula e SALVA no session_state.route
        if gerar:
            coords = [(float(lat), float(lon)) for lat, lon in zip(pontos_validos["lat"], pontos_validos["lon"])]
            ordem = construir_ordem(coords, metodo_ordem)
            coords_ordenadas = [coords[i] for i in ordem]
            if fechar_rota and len(coords_ordenadas) >= 2:
                coords_ordenadas.append(coords_ordenadas[0])

            route_coords_lonlat = None
            distancia_km_aprox = None
            duracao_min = None

            try:
                if motor_rota == "OSRM público (gratuito)":
                    geometry, dist_m, dur_s = osrm_route(coords_ordenadas, profile=perfil_osrm)
                    route_coords_lonlat = geometry
                    distancia_km_aprox = (dist_m / 1000.0) if dist_m is not None else None
                    duracao_min = (dur_s / 60.0) if dur_s is not None else None
                else:
                    route_coords_lonlat = [[lon, lat] for (lat, lon) in coords_ordenadas]
                    distancia_km_aprox = rota_distancia_km(coords, ordem, fechar_rota)
                    duracao_min = None
            except Exception as e:
                st.error(f"Falha ao calcular a rota: {e}")
                route_coords_lonlat = None

            ordem_df = pontos_validos.iloc[ordem].copy().reset_index(drop=True)
            ordem_df.index = ordem_df.index + 1
            ordem_df.rename_axis("Ordem", inplace=True)

            st.session_state.route = {
                "coords_lonlat": route_coords_lonlat,
                "ordem_df": ordem_df,
                "distancia_km": distancia_km_aprox,
                "duracao_min": duracao_min,
                "perfil": (perfil_osrm if motor_rota == "OSRM público (gratuito)" else "geodesic"),
                "fechar_rota": fechar_rota,
            }
            st.success("Rota gerada! Role para ver o mapa e os downloads.")

        # Renderização baseada no que está salvo em session_state.route
        route_state = st.session_state.route
        if route_state and route_state.get("coords_lonlat"):
            mostrar_mapa(pontos_validos, route_coords_lonlat=route_state["coords_lonlat"])

            st.subheader("Ordem dos clientes na rota")
            ordem_df = route_state["ordem_df"]
            st.dataframe(ordem_df[["cliente", "endereco", "lat", "lon", "status"]], use_container_width=True)

            colm1, colm2, colm3 = st.columns(3)
            if route_state.get("distancia_km") is not None:
                colm1.metric("Distância total", f"{route_state['distancia_km']:,.2f} km")
            if route_state.get("duracao_min") is not None:
                colm2.metric("Tempo estimado", f"{route_state['duracao_min']:,.1f} min")
            colm3.metric("Pontos na rota", f"{len(ordem_df)}")

            # Downloads
            csv_ordem = ordem_df.reset_index().rename(columns={"index": "Ordem"}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar CSV com a ordem da rota",
                data=csv_ordem,
                file_name="rota_clientes.csv",
                mime="text/csv",
            )

            try:
                geojson_str = montar_geojson_rota(
                    route_coords_lonlat=route_state["coords_lonlat"],
                    pontos_ordem_df=ordem_df,
                    distancia_km=route_state.get("distancia_km"),
                    duracao_min=route_state.get("duracao_min"),
                    profile=route_state.get("perfil"),
                    fechar_rota=route_state.get("fechar_rota", False),
                )

                st.download_button(
                    "⬇️ Baixar GeoJSON da rota",
                    data=geojson_str.encode("utf-8"),
                    file_name="rota_clientes.geojson",
                    mime="application/geo+json",
                )

                with st.expander("Prévia do GeoJSON (até ~2000 chars)"):
                    st.code(geojson_str[:2000], language="json")
            except Exception as e:
                st.error(f"Não foi possível gerar o GeoJSON: {e}")
        else:
            # Sem rota gerada, apenas mostra os pontos geocodificados
            mostrar_mapa(df_geo)

    # Resultado completo (geocodificação)
    st.subheader("Resultado (com latitude/longitude)")
    st.dataframe(df_geo, use_container_width=True)
    csv = df_geo.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV geocodificado", data=csv, file_name="clientes_geocodificados.csv", mime="text/csv")
else:
    st.info("Envie um arquivo e clique em **Geocodificar** para começar.")

# Rodapé
st.caption(
    "Dicas: para grandes volumes, use chave de API (Google/OpenCage) e/ou pré-geocodifique. "
    "No Nominatim, respeite limites (≈1 req/s) e inclua um user_agent com e-mail. "
    "Para rotas, o OSRM público é ótimo para testes; para produção, considere hospedar um OSRM próprio."
)