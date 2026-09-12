# Twilight Struggle: Complete 84-Country Map Topology Specification

This document provides the exhaustive country-by-country graph topology of the Deluxe Edition board (84 countries + 2 superpower nodes). It cross-references [`map.json`](map.json) and [`render_map.py`](render_map.py).

---

## 1. Regional Overview & Battleground Distribution

| Region | Region ID | Total Countries | Battlegrounds | Scoring Card ID | Subregions | Geopolitical Anomalies |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Europe** | $0$ | 21 | 5 | #2 (Europe Scoring) | Western Europe (14), Eastern Europe (9) *(Austria & Finland in both)* | Canada, Turkey are in Europe |
| **Middle East** | $2$ | 10 | 6 | #3 (Middle East Scoring) | — | Libya, Egypt are in Middle East |
| **Asia** | $1$ | 15 | 6 | #1 (Asia Scoring), #38 (SE Asia) | Southeast Asia (7) | — |
| **Africa** | $3$ | 18 | 5 | #78 (Africa Scoring) | — | — |
| **Central America** | $4$ | 10 | 3 | #37 (Central America Scoring) | — | — |
| **South America** | $5$ | 10 | 4 | #39 (South America Scoring) | — | — |
| **TOTAL** | — | **84** | **29** | — | — | — |

---

## 2. Complete 84-Country Directory

### 2.1 Europe (21 Countries, 5 Battlegrounds)

| ID | Country Name | Stability | Battleground | Subregions | Superpower Adjacency | Adjacent Neighbors |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **0** | **Canada** | 4 | No | Western Europe | **USA** | United Kingdom |
| **1** | **United Kingdom** | 5 | No | Western Europe | — | Canada, Norway, Benelux, France |
| **2** | **Norway** | 4 | No | Western Europe | — | United Kingdom, Sweden |
| **3** | **Sweden** | 4 | No | Western Europe | — | Norway, Denmark, Finland |
| **4** | **Denmark** | 3 | No | Western Europe | — | Sweden, West Germany |
| **5** | **Finland** | 4 | No | Western & Eastern Europe | **USSR** | Sweden |
| **6** | **Benelux** | 3 | No | Western Europe | — | United Kingdom, West Germany |
| **7** | **West Germany** | 4 | **YES** | Western Europe | — | Benelux, Denmark, France, Austria, East Germany |
| **8** | **France** | 3 | **YES** | Western Europe | — | United Kingdom, West Germany, Spain/Portugal, Italy, Algeria |
| **9** | **Spain/Portugal** | 2 | No | Western Europe | — | France, Italy, Morocco |
| **10** | **Italy** | 2 | **YES** | Western Europe | — | France, Spain/Portugal, Austria, Yugoslavia, Greece |
| **11** | **Greece** | 2 | No | Western Europe | — | Italy, Yugoslavia, Bulgaria, Turkey |
| **12** | **Turkey** | 2 | No | Western Europe | — | Greece, Bulgaria, Romania, Syria |
| **13** | **Austria** | 4 | No | Western & Eastern Europe | — | West Germany, East Germany, Italy, Hungary |
| **14** | **East Germany** | 3 | **YES** | Eastern Europe | — | West Germany, Austria, Poland, Czechoslovakia |
| **15** | **Poland** | 3 | **YES** | Eastern Europe | **USSR** | East Germany, Czechoslovakia |
| **16** | **Czechoslovakia** | 3 | No | Eastern Europe | — | East Germany, Poland, Hungary |
| **17** | **Hungary** | 3 | No | Eastern Europe | — | Austria, Czechoslovakia, Yugoslavia, Romania |
| **18** | **Yugoslavia** | 3 | No | Eastern Europe | — | Italy, Hungary, Romania, Greece |
| **19** | **Romania** | 3 | No | Eastern Europe | **USSR** | Hungary, Yugoslavia, Turkey |
| **20** | **Bulgaria** | 3 | No | Eastern Europe | — | Greece, Turkey |

---

### 2.2 Middle East (10 Countries, 6 Battlegrounds)

| ID | Country Name | Stability | Battleground | Superpower Adjacency | Adjacent Neighbors |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **21** | **Lebanon** | 1 | No | — | Syria, Israel, Jordan |
| **22** | **Syria** | 2 | No | — | Turkey, Lebanon, Israel |
| **23** | **Israel** | 4 | **YES** | — | Lebanon, Syria, Jordan, Egypt |
| **24** | **Iraq** | 3 | **YES** | — | Jordan, Saudi Arabia, Gulf States, Iran |
| **25** | **Iran** | 2 | **YES** | — | Iraq, Afghanistan, Pakistan |
| **26** | **Jordan** | 2 | No | — | Israel, Lebanon, Iraq, Saudi Arabia |
| **27** | **Gulf States** | 3 | No | — | Iraq, Saudi Arabia |
| **28** | **Saudi Arabia** | 3 | **YES** | — | Jordan, Iraq, Gulf States |
| **29** | **Egypt** | 2 | **YES** | — | Israel, Libya, Sudan |
| **30** | **Libya** | 2 | **YES** | — | Egypt, Tunisia |

---

### 2.3 Asia (15 Countries, 6 Battlegrounds)

| ID | Country Name | Stability | Battleground | Subregions | Superpower Adjacency | Adjacent Neighbors |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **31** | **Afghanistan** | 2 | No | — | **USSR** | Iran, Pakistan |
| **32** | **Pakistan** | 2 | **YES** | — | — | Iran, Afghanistan, India |
| **33** | **India** | 3 | **YES** | — | — | Pakistan, Burma |
| **34** | **Burma** | 2 | No | Southeast Asia | — | India, Laos/Cambodia |
| **35** | **Laos/Cambodia** | 1 | No | Southeast Asia | — | Burma, Thailand, Vietnam |
| **36** | **Thailand** | 2 | **YES** | Southeast Asia | — | Laos/Cambodia, Vietnam, Malaysia |
| **37** | **Vietnam** | 1 | No | Southeast Asia | — | Laos/Cambodia, Thailand |
| **38** | **Malaysia** | 2 | No | Southeast Asia | — | Thailand, Indonesia, Australia |
| **39** | **Indonesia** | 1 | No | Southeast Asia | — | Malaysia, Philippines |
| **40** | **Philippines** | 2 | No | Southeast Asia | — | Indonesia, Japan |
| **41** | **Australia** | 4 | No | — | — | Malaysia |
| **42** | **Taiwan** | 3 | No | — | — | Japan, South Korea |
| **43** | **North Korea** | 3 | **YES** | — | **USSR** | South Korea |
| **44** | **South Korea** | 3 | **YES** | — | — | North Korea, Japan, Taiwan |
| **45** | **Japan** | 4 | **YES** | — | **USA** | South Korea, Taiwan, Philippines |

---

### 2.4 Africa (18 Countries, 5 Battlegrounds)

| ID | Country Name | Stability | Battleground | Superpower Adjacency | Adjacent Neighbors |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **46** | **Morocco** | 3 | No | — | Spain/Portugal, Algeria, West African States |
| **47** | **Algeria** | 2 | **YES** | — | France, Morocco, Tunisia, Saharan States |
| **48** | **Tunisia** | 2 | No | — | Libya, Algeria |
| **49** | **West African States** | 2 | No | — | Morocco, Ivory Coast |
| **50** | **Saharan States** | 1 | No | — | Algeria, Nigeria |
| **51** | **Sudan** | 1 | No | — | Egypt, Ethiopia |
| **52** | **Ivory Coast** | 2 | No | — | West African States, Nigeria |
| **53** | **Nigeria** | 1 | **YES** | — | Saharan States, Ivory Coast, Cameroon |
| **54** | **Ethiopia** | 1 | No | — | Sudan, Somalia |
| **55** | **Somalia** | 2 | No | — | Ethiopia, Kenya |
| **56** | **Cameroon** | 1 | No | — | Nigeria, Zaire |
| **57** | **Zaire** | 1 | **YES** | — | Cameroon, Angola, Zimbabwe |
| **58** | **Kenya** | 2 | No | — | Somalia, SE African States |
| **59** | **Angola** | 1 | **YES** | — | Zaire, Botswana, South Africa |
| **60** | **Zimbabwe** | 1 | No | — | Zaire, SE African States, Botswana |
| **61** | **SE African States** | 1 | No | — | Kenya, Zimbabwe |
| **62** | **Botswana** | 2 | No | — | Angola, Zimbabwe, South Africa |
| **63** | **South Africa** | 3 | **YES** | — | Angola, Botswana |

---

### 2.5 Central America (10 Countries, 3 Battlegrounds)

| ID | Country Name | Stability | Battleground | Superpower Adjacency | Adjacent Neighbors |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **64** | **Mexico** | 2 | **YES** | **USA** | Guatemala |
| **65** | **Guatemala** | 1 | No | — | Mexico, El Salvador, Honduras |
| **66** | **El Salvador** | 1 | No | — | Guatemala, Honduras |
| **67** | **Honduras** | 2 | No | — | Guatemala, El Salvador, Nicaragua, Costa Rica |
| **68** | **Costa Rica** | 3 | No | — | Honduras, Nicaragua, Panama |
| **69** | **Nicaragua** | 1 | No | — | Honduras, Costa Rica, Cuba |
| **70** | **Panama** | 2 | **YES** | — | Costa Rica, Colombia |
| **71** | **Cuba** | 3 | **YES** | **USA** | Nicaragua, Haiti |
| **72** | **Haiti** | 1 | No | — | Cuba, Dominican Rep |
| **73** | **Dominican Rep** | 1 | No | — | Haiti |

---

### 2.6 South America (10 Countries, 4 Battlegrounds)

| ID | Country Name | Stability | Battleground | Superpower Adjacency | Adjacent Neighbors |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **74** | **Colombia** | 1 | No | — | Panama, Ecuador, Venezuela |
| **75** | **Ecuador** | 2 | No | — | Colombia, Peru |
| **76** | **Peru** | 2 | No | — | Ecuador, Chile, Bolivia |
| **77** | **Venezuela** | 2 | **YES** | — | Colombia, Brazil |
| **78** | **Bolivia** | 2 | No | — | Peru, Paraguay |
| **79** | **Brazil** | 2 | **YES** | — | Venezuela, Uruguay |
| **80** | **Paraguay** | 2 | No | — | Bolivia, Argentina, Uruguay |
| **81** | **Chile** | 3 | **YES** | — | Peru, Argentina |
| **82** | **Argentina** | 2 | **YES** | — | Chile, Paraguay, Uruguay |
| **83** | **Uruguay** | 2 | No | — | Brazil, Paraguay, Argentina |

---

## 3. Superpower Adjacency Registry

- **USA Connected Nodes (4 countries)**:
  1. Canada (Europe / Western Europe)
  2. Japan (Asia)
  3. Mexico (Central America)
  4. Cuba (Central America)

- **USSR Connected Nodes (6 countries)**:
  1. Finland (Europe / Western & Eastern Europe)
  2. Poland (Europe / Eastern Europe)
  3. Romania (Europe / Eastern Europe)
  4. Afghanistan (Asia)
  5. North Korea (Asia)
