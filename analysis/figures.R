estimate_detection_metrics <- function(df, recall = 0.63) {
  df %>%
    left_join(city_data) %>%
    transmute(
      city,
      type,
      period,
      road_network_length_km ,
      m_per_pano,
      pop_pano = 2 * road_network_length_km * 1000 / m_per_pano, # N
      n_pano, 
      n_detection, 
      # detection rate (unadjusted detections per pano)
      p_hat = n_detection / n_pano,
      # infinite population sd:
      p_hat_sd = sqrt(p_hat * (1 - p_hat) / n_pano),
      # for finite population sd:
      # p_hat_sd = sqrt((p_hat * (1 - p_hat) / n_pano) * ((pop_pano - n_pano) / (pop_pano - 1))),
      # detection rate (detections per km counting both sides of the road per km)
      est_detections_per_km = p_hat * (1000 / m_per_pano) * (2 / recall),
      se_detections_per_km = p_hat_sd * (1000 / m_per_pano) * (2 / recall),
      # detection count
      est_detections = est_detections_per_km * road_network_length_km,
      se_detections = se_detections_per_km * road_network_length_km
    ) %>%
    ungroup() %>%
    select(-p_hat, -p_hat_sd)
}

plot_camera_density <- function(df, legend = TRUE) {
  if (legend) {
    legend_position = "bottom"
  } else {
    legend_position = "none"
  }
  
  df %>%
    ggplot(aes(x = city, y = est_detections_per_km, fill = type)) +
    geom_col() +
    geom_linerange(aes(
      ymin = est_detections_per_km - 1.96*se_detections_per_km, 
      ymax = est_detections_per_km + 1.96*se_detections_per_km
    )) +
    scale_x_discrete(name = "") +
    scale_y_continuous(
      name = "Estimated cameras per km", 
      position = "right",
      expand = expansion(mult = c(0, 0.1))
    ) +
    scale_fill_discrete(name = "") +
    coord_flip() +
    theme(
      panel.border = element_blank(), 
      axis.line = element_line(size = 1, color = "black"),
      axis.title.x = element_text(family = "Helvetica", color = "black"), 
      axis.text = element_text(family = "Helvetica", color = "black"),
      legend.position = legend_position,
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    ) 
}

load_road_network <- function(city_name){
  stopifnot(city_name %in% city_data$city)
  
  path <- here::here("data", "road_network", city_name, "edges.shp")
  read_sf(path)
}

get_max_points <- function(df){
  df %>%
    select(geometry) %>%
    st_cast("POINT") %>%
    st_coordinates() %>%
    as_tibble() %>%
    summarize(
      x_max = max(X),
      x_min = min(X),
      y_max = max(Y),
      y_min = min(Y)
    ) 
}

generate_sampled_point_map <- function(df, city_name){
  # load road network
  road_network <- load_road_network(city_name)
  
  # get crs
  road_network_crs <- st_crs(road_network) %>%
    as.integer() 
  road_network_crs <- road_network_crs[1]
  
  # find bounding coordinates of road network
  bbox <- st_bbox(road_network)
  
  # plot points
  road_network %>%
    ggplot() +
    geom_sf(fill = "white", color = "gray", alpha = 0.6) +
    geom_sf(
      data = df %>%
        filter(city == city_name) %>%
        st_as_sf(coords = c("lon", "lat"),
                 # ensure same crs as road network
                 crs = road_network_crs, 
                 agr = "constant"), 
      color = "blue", size = 0.2, 
      shape = 16, alpha = 1 
    ) + 
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
    coord_sf(xlim = c(bbox$xmin, bbox$xmax), ylim = c(bbox$ymin, bbox$ymax)) +
    theme(
      axis.text = element_blank(), 
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 20)
    )
}

generate_detected_point_map <- function(df, city_name){
  # load road network
  road_network <- load_road_network(city_name)
  
  # get crs
  road_network_crs <- st_crs(road_network) %>%
    as.integer() 
  road_network_crs <- road_network_crs[1]
  
  # find bounding coordinates of road network
  bbox <- st_bbox(road_network)
  
  # plot points
  road_network %>%
    ggplot() +
    geom_sf(fill = "white", color = "gray", alpha = 0.6) +
    geom_sf(
      data = df %>%
        filter(
          city == city_name,
          camera_count > 0
        ) %>%
        st_as_sf(coords = c("lon", "lat"),
                 # ensure same crs as road network
                 crs = road_network_crs, 
                 agr = "constant"), 
      color = "red", size = 0.5,
      shape = 16, alpha = 1
    ) + 
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
    coord_sf(xlim = c(bbox$xmin, bbox$xmax), ylim = c(bbox$ymin, bbox$ymax)) +
    theme(
      axis.text = element_blank(), 
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 20)
    )
}

annotate_points_with_census <- function(df, city_name, census_var){
  stopifnot(census_var %in% c("income", "race"))
  
  # define state, county using `city_data`
  state <- city_data %>%
    filter(city == city_name) %>%
    pull(state)
  county <- city_data %>%
    filter(city == city_name) %>%
    pull(county)
  
  
  # specify variables
  summary_vars <- "B03002_001" # total population
  if (census_var == "income") {
    vars <- c(Income = "B19113_001")
  } else if (census_var == "race") {
    vars <- c(White = "B03002_003") #non-Hispanic white
  }
  
  # get census data
  if (city_name == "New York") {
    state = "NY"
    counties <- c("New York County", "Kings County", "Queens County", 
                  "Bronx County", "Richmond County")
    
    new_york <- purrr::map(
      counties,
      ~ get_acs(
        state = state, 
        county = .x, 
        geography = "block group",
        variables = vars,
        summary_var = summary_vars,
        geometry = TRUE
      )
    )
    
    df_census_block_group <- bind_rows(new_york)
    
  } else{
    
    if (city_name == "Washington") {
      county <- NULL
    }
    
    df_census_block_group <- get_acs(
      state = state, 
      county = county, 
      geography = "block group",
      variables = vars,
      summary_var = summary_vars,
      geometry = TRUE
    )
  }
  
  
  # add GIS features
  df <- df %>%
    filter(city == city_name) %>%
    # ensure same coords as tidycensus
    st_as_sf(
      coords = c("lon", "lat"),
      crs = 4269, 
      agr = "constant"
    )
  
  # annotate points with census data
  if (census_var == "income") {
    df <- st_join(
      df, 
      df_census_block_group %>% 
        select(GEOID, NAME, median_household_income = estimate, geometry)
    )
  } else if (census_var == "race") {
    df <- st_join(
      df, 
      df_census_block_group %>% 
        transmute(
          GEOID, NAME, 
          percentage_minority = (summary_est - estimate) / summary_est, geometry
        )
    )
  }
  
  df
}
