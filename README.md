# About Physics-contrained ET Hybrid Model

The hybrid model consists of two main components: a machine learning module for simulating surface resistance (rs) or aerodynamic resistance (ra), and a physical model for predicting LE.

We developed our hybrid models using Python. Similar to machine learning models, the dataset was partitioned into training (70%) and validation (20%) sets to dynamically update the internal model parameters during the fitting process. The test set (10%) was then used to assess the performance of the physics-constrained hybrid models. 

## Physics-based Models
The partitioning of energy at the Earth’s surface is governed by the following three coupled equations (Mu et al., 2007):
H=ρc_p  (T_s-T_a)/r_a                          …(1)
LE=(ρc_p)/γ  (e_s-e_a)/(r_a+r_s )                         …(2)
R_n-G=H+LE                      …(3)
Where H, LE are the fluxes of sensible heat (W m-2), latent heat (W m-2), Rn is net radiation (W m-2), G is soil heat flux (W m-2); Ts, Ta are the temperature of land surface and air (K); es, ea are saturation and actual vapour pressure (Pa); ra is the aerodynamic resistance (s m-1), rs is the surface resistance (s m-1), ρ is air density (kg m-3), and cp is the specific heat capacity of air (J kg-1 K-1), γ is the psychrometric constant (Pa K-1).

These fundamental equations constitute the foundation of various terrestrial ET modeling approaches. Accurate estimation of parameters—ra, rs, and radiative surface temperature—is essential for effective ET modeling (Chen and Liu, 2020). After reviewing the required model attributes for regional ET algorithms, we utilized two classical models: the surface energy balance model and Penman Monteith model.
2.3.1 Surface Energy Balance Model
The surface energy balance (SEB) model calculates the flux of sensible heat from Eq. (1) by calculating the aerodynamic resistance (ra) from:
r_a=1/(k^2 u) ln((Z_m-d)/Z_0m )ln((Z_h-d)/Z_0h )                      …(4)
Where k is von Karman’s constant (0.4), u is wind speed (m s-1) at the measurement height (Zm) (m), Zh is height of humidity measurements (m), d is zero plane displacement height (m), Z0m is roughness length governing momentum transfer (m), Z0h is roughness length governing transfer of heat and vapour (m). The quantities d, Z0m, and Z0h are estimated as 2h/3, 0.1h, and 0.01h, respectively, where h is canopy height (m). Zm, Zh, and h are site-specific. LE is then calculated as the residual of the energy balance using Eq. (5). SEB mainly requires Tair, Tsoil, P, WS, Rn, G, hc.
LE=R_n-G-ρc_p  (T_s-T_a)/r_a                       …(5)
### Penman Monteith Model
The Penman Monteith (PM) model has been successfully applied to the MODIS global ET framework, which eliminates Ts (Eq. (6)) and utilizes remotely sensed vegetation indices to inform the model of the water availability on the land surface (Eq. (7)).
e_s=e_a+∆(T_s-T_a)                      …(6)
LE=(Δ(R_n-G)+ρc_P (e_s-e_a)/r_a)/(Δ+γ(1+r_s/r_a))                  …(7)
Where ∆ is the slope of the saturated vapor pressure curve (kPa K-1), The surface resistances (rs) used in MODIS ET for different plant function types is determined by the Biome Properties Look-Up Table (BPLUT) (Mu et al., 2007; Mu et al., 2011). PM mainly requires Tair, Tair_min, RH, P, WS, Rn, G, hc, LAI.

