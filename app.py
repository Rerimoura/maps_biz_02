# -*- coding: utf-8 -*-
# App Streamlit: subir clientes + endereços, visualizar no mapa, traçar rota com ponto de partida/destino e exportar GeoJSON/KML/KMZ
# Autor: Você :)
# Observação: comentários e rótulos em PT-BR para facilitar manutenção

import math
import json
import io
import zipfile
from html import escape

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
st.set_page_config(page_title="Clientes no Mapa + Rotas + GeoJSON/KML", layout="wide")

st.title("📍 Clientes no Mapa + 🛣️ Rotas + 🌐 GeoJSON/KML")
st.caption(
    "Envie uma planilha (CSV/XLSX) com **Cliente** e **Endereço**, geocodifique, visualize no mapa, "
    "defina **ponto de partida** e **ponto de destino**, trace rota (linhas retas ou OSRM) e exporte **GeoJSON/KML/KMZ**."
)

# ==============================
# Estado (session_state) helpers
# ==============================
def _init_state():
    if "df_geo" not in st.session_state:
        st.session_state.df_geo = None
    if "route" not in st.session_state:
        # dict: coords_lonlat, ordem_df, distancia_km, duracao_min, perfil, fechar, partida, destino
        st.session_state.route = None
    if "start" not in st.session_state:
        # dict: {"lat": float, "lon": float, "label": str}
        st.session_state.start = None
    if "end" not in st.session_state:
        # dict: {"lat": float, "lon": float, "label": str}
        st.session_state.end = None

def reset_geocoded():
    st.session_state.df_geo = None
    st.session_state.route = None
    st.session_state.start = None
    st.session_state.end = None

def reset_route():
    st.session_state.route = None

def reset_start():
    st.session_state.start = None

def reset_end():
    st.session_state.end = None

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


def rota_distancia_clientes_km(coords: List[Tuple[float, float]], ordem: List[int]) -> float:
    """Distância total (km) apenas ENTRE os clientes na ordem (rota aberta)."""
    if len(ordem) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(ordem) - 1):
        a, b = ordem[i], ordem[i+1]
        dist += haversine_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
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
    """Melhora a ordem (rota aberta) usando 2-opt básico mantendo o primeiro fixo."""
    if len(ordem) < 4:
        return ordem[:]
    best = ordem[:]
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, len(best) - 2):  # começa em 1 para manter início fixo
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


def construir_ordem(coords: List[Tuple[float, float]], metodo: str, start_index: int = 0) -> List[int]:
    """Constroi ordem conforme método e ponto de partida (start_index)."""
    n = len(coords)
    if n == 0:
        return []
    start_index = int(start_index) if 0 <= int(start_index) < n else 0

    if metodo == "Pela ordem do arquivo":
        ordem = list(range(n))
        # rotaciona para começar no start_index
        if start_index > 0:
            ordem = ordem[start_index:] + ordem[:start_index]
        return ordem
    elif metodo == "Otimizar (Vizinho + 2-opt)":
        nn = nearest_neighbor_order(coords, start_index=start_index)  # já começa no start_index
        return two_opt(coords, nn, max_iter=200)                      # mantém início fixo
    else:
        return list(range(n))


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
# GeoJSON: LineString (rota) + Points (waypoints + partida/destino)
# ==============================
def montar_geojson_rota(
    route_coords_lonlat,
    pontos_ordem_df: pd.DataFrame,
    distancia_km: Optional[float] = None,
    duracao_min: Optional[float] = None,
    profile: Optional[str] = None,
    fechar_rota: bool = False,
    ponto_partida: Optional[dict] = None,   # {"lat": float, "lon": float, "label": str}
    ponto_destino: Optional[dict] = None,   # {"lat": float, "lon": float, "label": str}
) -> str:
    """
    Monta um FeatureCollection GeoJSON com:
      - 1 LineString: a rota (coordinates em [lon, lat])
      - N Points: waypoints na ordem da rota
      - 1 Point opcional: partida
      - 1 Point opcional: destino
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
            "geometry": {"type": "LineString", "coordinates": route_coords_lonlat}
        })

    # Partida
    if ponto_partida and "lat" in ponto_partida and "lon" in ponto_partida:
        try:
            plat = float(ponto_partida["lat"])
            plon = float(ponto_partida["lon"])
            plabel = str(ponto_partida.get("label", "Partida"))
            features.append({
                "type": "Feature",
                "properties": {"tipo": "partida", "nome": plabel},
                "geometry": {"type": "Point", "coordinates": [plon, plat]}
            })
        except Exception:
            pass

    # Destino
    if ponto_destino and "lat" in ponto_destino and "lon" in ponto_destino:
        try:
            dlat = float(ponto_destino["lat"])
            dlon = float(ponto_destino["lon"])
            dlabel = str(ponto_destino.get("label", "Destino"))
            features.append({
                "type": "Feature",
                "properties": {"tipo": "destino", "nome": dlabel},
                "geometry": {"type": "Point", "coordinates": [dlon, dlat]}
            })
        except Exception:
            pass

    # Pontos dos clientes
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
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })

    fc = {"type": "FeatureCollection", "features": features}
    return json.dumps(fc, ensure_ascii=False, indent=2)


# ==============================
# KML/KMZ: LineString (rota) + Placemarks (waypoints + partida/destino)
# ==============================
def montar_kml_rota(
    route_coords_lonlat,
    pontos_ordem_df: pd.DataFrame,
    distancia_km: Optional[float] = None,
    duracao_min: Optional[float] = None,
    profile: Optional[str] = None,
    fechar_rota: bool = False,
    ponto_partida: Optional[dict] = None,   # {"lat": float, "lon": float, "label": str}
    ponto_destino: Optional[dict] = None,   # {"lat": float, "lon": float, "label": str}
) -> str:
    """
    Gera um KML (texto) com:
      - LineString da rota (se houver)
      - Ponto de partida (se houver)
      - Ponto de destino (se houver)
      - Pontos dos clientes na ordem
    """
    def _fmt(v, nd=6):
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return ""

    # Cabeçalho e estilos (KML usa cor ARGB em hex: aabbggrr)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Rota de Clientes</name>",
        "<description><![CDATA[",
        f"Perfil: {escape(str(profile or 'geodesic'))} | "
        f"Distância: {'' if distancia_km is None else f'{distancia_km:.2f} km'} | "
        f"Duração: {'' if duracao_min is None else f'{duracao_min:.1f} min'} | "
        f"Rota fechada: {'Sim' if fechar_rota else 'Não'}",
        "]]></description>",
        # Estilo da rota (vermelho semi-transparente)
        '<Style id="routeStyle"><LineStyle><color>990000ff</color><width>4</width></LineStyle></Style>',
        # Estilo dos pontos (clientes) – laranja
        '<Style id="pointStyle"><IconStyle><color>ff0099FF</color></IconStyle></Style>',
        # Estilo do ponto de partida – verde
        '<Style id="startStyle"><IconStyle><color>ff00cc00</color></IconStyle></Style>',
        # Estilo do ponto de destino – roxo
        '<Style id="endStyle"><IconStyle><color>ff9900ff</color></IconStyle></Style>',
    ]

    # Partida
    if ponto_partida and "lat" in ponto_partida and "lon" in ponto_partida:
        plat, plon = ponto_partida["lat"], ponto_partida["lon"]
        plabel = escape(str(ponto_partida.get("label", "Partida")))
        parts += [
            "<Placemark>",
            f"<name>{plabel}</name>",
            "<styleUrl>#startStyle</styleUrl>",
            "<Point>",
            f"<coordinates>{_fmt(plon)},{_fmt(plat)},0</coordinates>",
            "</Point>",
            "</Placemark>",
        ]

    # Destino
    if ponto_destino and "lat" in ponto_destino and "lon" in ponto_destino:
        dlat, dlon = ponto_destino["lat"], ponto_destino["lon"]
        dlabel = escape(str(ponto_destino.get("label", "Destino")))
        parts += [
            "<Placemark>",
            f"<name>{dlabel}</name>",
            "<styleUrl>#endStyle</styleUrl>",
            "<Point>",
            f"<coordinates>{_fmt(dlon)},{_fmt(dlat)},0</coordinates>",
            "</Point>",
            "</Placemark>",
        ]

    # Pontos dos clientes (na ordem)
    for idx, row in pontos_ordem_df.reset_index(drop=True).iterrows():
        try:
            lat = float(row["lat"]); lon = float(row["lon"])
        except Exception:
            continue
        nome = escape(f"{idx+1:02d} — {str(row.get('cliente',''))}")
        desc = escape(str(row.get("endereco", "")))
        parts += [
            "<Placemark>",
            f"<name>{nome}</name>",
            "<styleUrl>#pointStyle</styleUrl>",
            "<description><![CDATA[",
            f"{desc}",
            "]]></description>",
            "<Point>",
            f"<coordinates>{_fmt(lon)},{_fmt(lat)},0</coordinates>",
            "</Point>",
            "</Placemark>",
        ]

    # LineString da rota (se houver)
    if route_coords_lonlat and len(route_coords_lonlat) >= 2:
        coords_str = " ".join([f"{_fmt(lon)},{_fmt(lat)},0" for lon, lat in route_coords_lonlat])
        parts += [
            "<Placemark>",
            "<name>Rota</name>",
            "<styleUrl>#routeStyle</styleUrl>",
            "<LineString>",
            "<tessellate>1</tessellate>",
            f"<coordinates>{coords_str}</coordinates>",
            "</LineString>",
            "</Placemark>",
        ]

    parts += ["</Document>", "</kml>"]
    return "\n".join(parts)


def mostrar_mapa(df_geo: pd.DataFrame, route_coords_lonlat: Optional[List[List[float]]] = None,
                 partida: Optional[dict] = None, destino: Optional[dict] = None):
    """Renderiza os pontos e, opcionalmente, a rota e os pontos de partida/destino no mapa usando pydeck."""
    pontos = df_geo.dropna(subset=["lat", "lon"]).copy()
    if pontos.empty:
        st.warning("Nenhum ponto válido para plotar no mapa.")
        return

    mean_lat = float(pontos["lat"].astype(float).mean())
    mean_lon = float(pontos["lon"].astype(float).mean())

    layers = []

    # Pontos dos clientes
    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=pontos,
        get_position="[lon, lat]",
        get_fill_color=[0, 122, 255, 200],
        get_radius=80,
        pickable=True,
    )
    layers.append(layer_points)

    # Rota
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

    # Ponto de partida (se houver)
    if partida and "lat" in partida and "lon" in partida:
        partida_df = pd.DataFrame([{"lat": partida["lat"], "lon": partida["lon"], "label": partida.get("label", "Partida")}])
        layer_start = pdk.Layer(
            "ScatterplotLayer",
            data=partida_df,
            get_position="[lon, lat]",
            get_fill_color=[0, 180, 0, 220],  # verde
            get_radius=120,
            pickable=True,
        )
        layers.append(layer_start)

    # Ponto de destino (se houver)
    if destino and "lat" in destino and "lon" in destino:
        destino_df = pd.DataFrame([{"lat": destino["lat"], "lon": destino["lon"], "label": destino.get("label", "Destino")}])
        layer_end = pdk.Layer(
            "ScatterplotLayer",
            data=destino_df,
            get_position="[lon, lat]",
            get_fill_color=[170, 0, 255, 220],  # roxo
            get_radius=120,
            pickable=True,
        )
        layers.append(layer_end)

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
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("🧹 Limpar cache de geocodificação"):
            st.cache_data.clear()
            st.success("Cache limpo!")
    with col_b:
        if st.button("♻️ Limpar resultados"):
            reset_geocoded()
            st.success("Resultados limpos!")
    with col_c:
        if st.button("🧭 Limpar partida"):
            reset_start()
            st.success("Ponto de partida limpo!")
    with col_d:
        if st.button("🏁 Limpar destino"):
            reset_end()
            st.success("Ponto de destino limpo!")


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

    # Botão: Geocodificar (salva no state)
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
        st.session_state.df_geo = df_geo
        st.session_state.route = None
        st.success("Geocodificação concluída! Role até a seção de rotas.")

# ==============================
# Se existe geocodificação salva, mostra resultados e permite rotas
# ==============================
if st.session_state.df_geo is not None:
    df_geo = st.session_state.df_geo

    # Métricas
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
    # Seção: Rotas
    # ==============================
    st.subheader("🛣️ Rota entre clientes (opcional)")

    pontos_validos = df_geo.dropna(subset=["lat", "lon"]).reset_index(drop=True).copy()
    if len(pontos_validos) < 2:
        st.info("São necessários pelo menos **2** pontos geocodificados para traçar uma rota.")
        mostrar_mapa(df_geo)
    else:
        # ---- Ponto de partida ----
        st.markdown("**Ponto de partida**")
        colp1, colp2, colp3 = st.columns([1.2, 1, 1])
        with colp1:
            modo_partida = st.selectbox(
                "Escolha a origem",
                options=["Primeiro cliente", "Selecionar cliente da lista", "Endereço fixo (novo)"],
                index=0,
                help="Defina de onde a rota começa."
            )
        start_index = 0
        if modo_partida == "Selecionar cliente da lista":
            opcoes_idx = list(pontos_validos.index)
            def _fmt_start(i): 
                return f"{pontos_validos.loc[i,'cliente']} — {pontos_validos.loc[i,'endereco']}"
            with colp2:
                start_idx_sel = st.selectbox("Cliente de partida", options=opcoes_idx, format_func=_fmt_start)
                start_index = int(pontos_validos.index.get_loc(start_idx_sel))
        elif modo_partida == "Endereço fixo (novo)":
            with colp2:
                start_address = st.text_input("Endereço de partida (fixo)", placeholder="Ex.: Rua X, 123 - Cidade/UF, BR")
            with colp3:
                if st.button("Geocodificar partida", help="Geocodifica e salva este endereço como ponto de partida."):
                    if not start_address.strip():
                        st.warning("Informe um endereço de partida.")
                    else:
                        plat, plon, pstatus = geocodificar_endereco_cache(
                            endereco=start_address.strip(),
                            provedor=provedor,
                            api_key=api_key,
                            country_bias=country_bias,
                            language=language,
                            user_agent_email=user_agent_email or None,
                        )
                        if pstatus == "OK":
                            st.session_state.start = {"lat": float(plat), "lon": float(plon), "label": start_address.strip()}
                            st.success("Ponto de partida salvo!")
                        else:
                            st.error(f"Não foi possível geocodificar a partida: {pstatus}")
        # Exibe partida salva (se houver)
        if st.session_state.start:
            st.info(f"Partida fixa: **{st.session_state.start.get('label','(sem nome)')}** "
                    f"({st.session_state.start['lat']:.6f}, {st.session_state.start['lon']:.6f})")

        # ---- Ponto de destino ----
        st.markdown("**Ponto de destino**")
        cold1, cold2, cold3 = st.columns([1.2, 1, 1])
        with cold1:
            modo_destino = st.selectbox(
                "Escolha o destino",
                options=["Último cliente", "Selecionar cliente da lista", "Endereço fixo (novo)"],
                index=0,
                help="Defina onde a rota termina."
            )
        end_index = None  # índice (0..n-1) do cliente destino, se selecionado
        if modo_destino == "Selecionar cliente da lista":
            opcoes_idx_d = list(pontos_validos.index)
            def _fmt_end(i): 
                return f"{pontos_validos.loc[i,'cliente']} — {pontos_validos.loc[i,'endereco']}"
            with cold2:
                end_idx_sel = st.selectbox("Cliente de destino", options=opcoes_idx_d, format_func=_fmt_end)
                end_index = int(pontos_validos.index.get_loc(end_idx_sel))
        elif modo_destino == "Endereço fixo (novo)":
            with cold2:
                end_address = st.text_input("Endereço de destino (fixo)", placeholder="Ex.: Rua Y, 999 - Cidade/UF, BR")
            with cold3:
                if st.button("Geocodificar destino", help="Geocodifica e salva este endereço como ponto de destino."):
                    if not end_address.strip():
                        st.warning("Informe um endereço de destino.")
                    else:
                        dlat, dlon, dstatus = geocodificar_endereco_cache(
                            endereco=end_address.strip(),
                            provedor=provedor,
                            api_key=api_key,
                            country_bias=country_bias,
                            language=language,
                            user_agent_email=user_agent_email or None,
                        )
                        if dstatus == "OK":
                            st.session_state.end = {"lat": float(dlat), "lon": float(dlon), "label": end_address.strip()}
                            st.success("Ponto de destino salvo!")
                        else:
                            st.error(f"Não foi possível geocodificar o destino: {dstatus}")
        # Exibe destino salvo (se houver)
        if st.session_state.end:
            st.info(f"Destino fixo: **{st.session_state.end.get('label','(sem nome)')}** "
                    f"({st.session_state.end['lat']:.6f}, {st.session_state.end['lon']:.6f})")

        # ---- Opções de rota ----
        col1, col2, col3 = st.columns(3)
        with col1:
            metodo_ordem = st.selectbox(
                "Método de ordenação",
                options=["Pela ordem do arquivo", "Otimizar (Vizinho + 2-opt)"],
                help="Como a ordem dos clientes será definida.",
                key="opt_metodo_ordem"
            )
        with col2:
            fechar_rota = st.checkbox("Voltar ao início (rota fechada)", value=False, key="opt_fechar")
        with col3:
            motor_rota = st.selectbox(
                "Motor de rota",
                options=["Linhas retas (sem API)", "OSRM público (gratuito)"],
                help="OSRM retorna distância/tempo reais; Linhas retas estimam distância geodésica.",
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

        # ---- Geração da rota (salva no session_state.route) ----
        if gerar:
            coords = [(float(lat), float(lon)) for lat, lon in zip(pontos_validos["lat"], pontos_validos["lon"])]

            # Ordem de visita começando no start_index (se partida é cliente)
            ordem = construir_ordem(coords, metodo_ordem, start_index=start_index)

            # Se destino é um cliente específico, garante que ele seja o último (sem duplicar)
            if modo_destino == "Selecionar cliente da lista" and end_index is not None:
                if end_index in ordem and end_index != ordem[-1]:
                    # remove da posição atual e envia para o fim (mantém start em primeiro)
                    ordem.remove(end_index)
                    ordem.append(end_index)

            # Coordenadas dos clientes na ordem
            coords_ordenadas_clientes = [coords[i] for i in ordem]

            # Partida/destino externos (endereços fixos)
            start_external = st.session_state.start if (modo_partida == "Endereço fixo (novo)" and st.session_state.start) else None
            end_external = st.session_state.end if (modo_destino == "Endereço fixo (novo)" and st.session_state.end) else None

            # Monta a sequência que irá para o motor de rota
            coords_para_rota = coords_ordenadas_clientes[:]
            if start_external:
                coords_para_rota = [(start_external["lat"], start_external["lon"])] + coords_para_rota
            if end_external:
                coords_para_rota = coords_para_rota + [(end_external["lat"], end_external["lon"])]

            if fechar_rota and len(coords_para_rota) >= 2:
                coords_para_rota = coords_para_rota + [coords_para_rota[0]]

            route_coords_lonlat = None
            distancia_km_aprox = None
            duracao_min = None

            try:
                if motor_rota == "OSRM público (gratuito)":
                    geometry, dist_m, dur_s = osrm_route(coords_para_rota, profile=perfil_osrm)
                    route_coords_lonlat = geometry
                    distancia_km_aprox = (dist_m / 1000.0) if dist_m is not None else None
                    duracao_min = (dur_s / 60.0) if dur_s is not None else None
                else:
                    # Linhas retas: polyline de tudo que será percorrido
                    route_coords_lonlat = [[lon, lat] for (lat, lon) in coords_para_rota]

                    # Distância aproximada:
                    # 1) entre clientes (rota aberta entre eles)
                    distancia_km_aprox = rota_distancia_clientes_km(coords, ordem)

                    # 2) add partida externa -> primeiro cliente
                    if start_external and len(coords_ordenadas_clientes) >= 1:
                        distancia_km_aprox += haversine_km(
                            start_external["lat"], start_external["lon"],
                            coords_ordenadas_clientes[0][0], coords_ordenadas_clientes[0][1],
                        )

                    # 3) add último cliente -> destino externo
                    if end_external and len(coords_ordenadas_clientes) >= 1:
                        distancia_km_aprox += haversine_km(
                            coords_ordenadas_clientes[-1][0], coords_ordenadas_clientes[-1][1],
                            end_external["lat"], end_external["lon"],
                        )

                    # 4) fechamento (fecha no primeiro da sequência usada no traçado)
                    if fechar_rota:
                        if start_external:
                            first_lat, first_lon = start_external["lat"], start_external["lon"]
                        else:
                            first_lat, first_lon = coords_ordenadas_clientes[0]
                        if end_external:
                            last_lat, last_lon = end_external["lat"], end_external["lon"]
                        else:
                            last_lat, last_lon = coords_ordenadas_clientes[-1]
                        distancia_km_aprox += haversine_km(last_lat, last_lon, first_lat, first_lon)

                    duracao_min = None
            except Exception as e:
                st.error(f"Falha ao calcular a rota: {e}")
                route_coords_lonlat = None

            # Tabela ordem (somente clientes na ordem)
            ordem_df = pontos_validos.iloc[ordem].copy().reset_index(drop=True)
            ordem_df.index = ordem_df.index + 1
            ordem_df.rename_axis("Ordem", inplace=True)

            # Salva no estado
            st.session_state.route = {
                "coords_lonlat": route_coords_lonlat,
                "ordem_df": ordem_df,
                "distancia_km": distancia_km_aprox,
                "duracao_min": duracao_min,
                "perfil": (perfil_osrm if motor_rota == "OSRM público (gratuito)" else "geodesic"),
                "fechar_rota": fechar_rota,
                "partida": start_external,   # None se partida for cliente
                "destino": end_external,     # None se destino for cliente
            }
            st.success("Rota gerada! Role para ver o mapa e os downloads.")

        # ---- Renderização com base no estado salvo ----
        route_state = st.session_state.route
        if route_state and route_state.get("coords_lonlat"):
            mostrar_mapa(
                pontos_validos,
                route_coords_lonlat=route_state["coords_lonlat"],
                partida=route_state.get("partida"),
                destino=route_state.get("destino"),
            )

            st.subheader("Ordem dos clientes na rota")
            ordem_df = route_state["ordem_df"]
            st.dataframe(ordem_df[["cliente", "endereco", "lat", "lon", "status"]], use_container_width=True)

            colm1, colm2, colm3 = st.columns(3)
            if route_state.get("distancia_km") is not None:
                colm1.metric("Distância total", f"{route_state['distancia_km']:,.2f} km")
            if route_state.get("duracao_min") is not None:
                colm2.metric("Tempo estimado", f"{route_state['duracao_min']:,.1f} min")
            colm3.metric("Pontos na rota", f"{len(ordem_df)}")

            # Downloads: CSV (ordem)
            csv_ordem = ordem_df.reset_index().rename(columns={"index": "Ordem"}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar CSV com a ordem da rota",
                data=csv_ordem,
                file_name="rota_clientes.csv",
                mime="text/csv",
            )

            # Downloads: GeoJSON
            try:
                geojson_str = montar_geojson_rota(
                    route_coords_lonlat=route_state["coords_lonlat"],
                    pontos_ordem_df=ordem_df,
                    distancia_km=route_state.get("distancia_km"),
                    duracao_min=route_state.get("duracao_min"),
                    profile=route_state.get("perfil"),
                    fechar_rota=route_state.get("fechar_rota", False),
                    ponto_partida=route_state.get("partida"),
                    ponto_destino=route_state.get("destino"),
                )
                st.download_button(
                    "⬇️ Baixar GeoJSON da rota",
                    data=geojson_str.encode("utf-8"),
                    file_name="rota_clientes.geojson",
                    mime="application/geo+json",
                )
            except Exception as e:
                st.error(f"Não foi possível gerar o GeoJSON: {e}")

            # Downloads: KML e KMZ
            try:
                kml_str = montar_kml_rota(
                    route_coords_lonlat=route_state["coords_lonlat"],
                    pontos_ordem_df=ordem_df,
                    distancia_km=route_state.get("distancia_km"),
                    duracao_min=route_state.get("duracao_min"),
                    profile=route_state.get("perfil"),
                    fechar_rota=route_state.get("fechar_rota", False),
                    ponto_partida=route_state.get("partida"),
                    ponto_destino=route_state.get("destino"),
                )

                st.download_button(
                    "⬇️ Baixar KML da rota",
                    data=kml_str.encode("utf-8"),
                    file_name="rota_clientes.kml",
                    mime="application/vnd.google-earth.kml+xml",
                )

                # KMZ (zip contendo doc.kml)
                kml_bytes = kml_str.encode("utf-8")
                kmz_buffer = io.BytesIO()
                with zipfile.ZipFile(kmz_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("doc.kml", kml_bytes)
                kmz_data = kmz_buffer.getvalue()

                st.download_button(
                    "⬇️ Baixar KMZ da rota",
                    data=kmz_data,
                    file_name="rota_clientes.kmz",
                    mime="application/vnd.google-earth.kmz",
                )

                with st.expander("Prévia do GeoJSON (até ~2000 chars)"):
                    st.code(geojson_str[:2000], language="json")

            except Exception as e:
                st.error(f"Não foi possível gerar KML/KMZ: {e}")
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