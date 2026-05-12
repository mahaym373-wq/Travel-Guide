"""
data.py — Semantic Network Data Layer
======================================
Smart India Travel Guide Chatbot

This module defines the raw relationship triples that populate the
semantic network.  Each triple is a (Subject, Predicate, Object) tuple
stored in the master list `RELATIONSHIPS`.

Design Principles
-----------------
* **Depth over breadth** — chains go 4–5 levels deep so the reasoning
  engine can perform multi-hop inference.
* **Bi-directional traceability** — although triples are stored in one
  direction, the graph-building layer can invert any edge
  (e.g. "Manali  →has→  Mountains" implies "Mountains  →found_in→  Manali").
* **PEP-8 compliant** — all identifiers, comments, and formatting follow
  PEP-8 guidelines.

No graph traversal or query logic lives here — only raw data.
"""

# ──────────────────────────────────────────────────────────────────────
#  DESTINATIONS  (20 Indian destinations)
# ──────────────────────────────────────────────────────────────────────
#  Goa, Manali, Jaipur, Kerala, Varanasi, Rishikesh, Ladakh, Udaipur,
#  Darjeeling, Agra, Shimla, Hampi, Andaman Islands, Munnar, Rann of Kutch,
#  Coorg, Ooty, Mysore, Amritsar, Jodhpur
# ──────────────────────────────────────────────────────────────────────


RELATIONSHIPS: list[tuple[str, str, str]] = [

    # ==================================================================
    #  1. GOA — Beach & Nightlife Chain
    # ==================================================================
    # Level 1: destination → feature
    ("Goa", "has", "Beaches"),
    ("Goa", "has", "Nightlife"),
    ("Goa", "has", "Portuguese Architecture"),
    ("Goa", "located_in", "West India"),
    ("Goa", "best_season", "Winter"),
    ("Goa", "budget_level", "Moderate Budget"),
    ("Goa", "known_for", "Seafood Cuisine"),
    ("Goa", "type", "Leisure Destination"),

    # Level 2: feature → activity
    ("Beaches", "enable", "Water Sports"),
    ("Beaches", "enable", "Sunbathing"),
    ("Nightlife", "enable", "Clubbing"),
    ("Portuguese Architecture", "enable", "Heritage Walks"),

    # Level 3: activity → requirement
    ("Water Sports", "requires", "Swimming Skills"),
    ("Water Sports", "type", "Adventure Activity"),
    ("Clubbing", "requires", "Nightlife Budget"),
    ("Heritage Walks", "requires", "Comfortable Footwear"),

    # Level 4: requirement → outcome
    ("Swimming Skills", "improves", "Physical Fitness"),
    ("Nightlife Budget", "part_of", "High Budget"),

    # Level 5: outcome → benefit
    ("Physical Fitness", "leads_to", "Better Health"),

    # ==================================================================
    #  2. MANALI — Mountain & Trekking Chain
    # ==================================================================
    ("Manali", "has", "Mountains"),
    ("Manali", "has", "Snow"),
    ("Manali", "has", "Rohtang Pass"),
    ("Manali", "located_in", "North India"),
    ("Manali", "best_season", "Summer"),
    ("Manali", "budget_level", "Moderate Budget"),
    ("Manali", "type", "Adventure Destination"),

    # Level 2
    ("Mountains", "enable", "Trekking"),
    ("Mountains", "enable", "Camping"),
    ("Snow", "enable", "Skiing"),
    ("Rohtang Pass", "type", "Mountain Pass"),

    # Level 3
    ("Trekking", "requires", "High Fitness"),
    ("Trekking", "type", "Adventure Activity"),
    ("Camping", "requires", "Camping Gear"),
    ("Skiing", "requires", "Skiing Equipment"),

    # Level 4
    ("High Fitness", "improves", "Better Health"),
    ("Camping Gear", "available_at", "Local Rental Shops"),

    # Level 5
    ("Better Health", "enables", "Longer Travel"),

    # ==================================================================
    #  3. JAIPUR — Heritage & Culture Chain
    # ==================================================================
    ("Jaipur", "has", "Forts"),
    ("Jaipur", "has", "Palaces"),
    ("Jaipur", "has", "Bazaars"),
    ("Jaipur", "located_in", "North India"),
    ("Jaipur", "best_season", "Winter"),
    ("Jaipur", "budget_level", "Low Budget"),
    ("Jaipur", "known_for", "Rajasthani Cuisine"),
    ("Jaipur", "type", "Cultural Destination"),
    ("Jaipur", "also_known_as", "Pink City"),

    # Level 2
    ("Forts", "enable", "History Tours"),
    ("Palaces", "enable", "Photography"),
    ("Bazaars", "enable", "Shopping"),

    # Level 3
    ("History Tours", "requires", "Guide Booking"),
    ("History Tours", "type", "Cultural Activity"),
    ("Photography", "requires", "Camera Equipment"),
    ("Shopping", "requires", "Bargaining Skills"),

    # Level 4
    ("Guide Booking", "costs", "Low Budget"),
    ("Bargaining Skills", "improves", "Negotiation Ability"),

    # Level 5
    ("Negotiation Ability", "useful_for", "Daily Life"),

    # ==================================================================
    #  4. KERALA — Backwaters & Ayurveda Chain
    # ==================================================================
    ("Kerala", "has", "Backwaters"),
    ("Kerala", "has", "Ayurveda Centers"),
    ("Kerala", "has", "Tea Plantations"),
    ("Kerala", "has", "Wildlife Sanctuaries"),
    ("Kerala", "located_in", "South India"),
    ("Kerala", "best_season", "Winter"),
    ("Kerala", "budget_level", "Moderate Budget"),
    ("Kerala", "known_for", "Kerala Cuisine"),
    ("Kerala", "type", "Wellness Destination"),
    ("Kerala", "also_known_as", "Gods Own Country"),

    # Level 2
    ("Backwaters", "enable", "Houseboat Cruises"),
    ("Ayurveda Centers", "enable", "Ayurvedic Treatment"),
    ("Tea Plantations", "enable", "Plantation Tours"),
    ("Wildlife Sanctuaries", "enable", "Wildlife Safari"),

    # Level 3
    ("Houseboat Cruises", "requires", "Advance Booking"),
    ("Ayurvedic Treatment", "requires", "Minimum 7 Days Stay"),
    ("Ayurvedic Treatment", "type", "Wellness Activity"),
    ("Wildlife Safari", "requires", "Forest Permit"),

    # Level 4
    ("Advance Booking", "costs", "Moderate Budget"),
    ("Minimum 7 Days Stay", "improves", "Deep Relaxation"),

    # Level 5
    ("Deep Relaxation", "leads_to", "Mental Wellness"),

    # ==================================================================
    #  5. VARANASI — Spiritual & Religious Chain
    # ==================================================================
    ("Varanasi", "has", "Ghats"),
    ("Varanasi", "has", "Temples"),
    ("Varanasi", "has", "Ganga River"),
    ("Varanasi", "located_in", "North India"),
    ("Varanasi", "best_season", "Winter"),
    ("Varanasi", "budget_level", "Low Budget"),
    ("Varanasi", "known_for", "Street Food"),
    ("Varanasi", "type", "Spiritual Destination"),
    ("Varanasi", "also_known_as", "City of Lights"),

    # Level 2
    ("Ghats", "enable", "Ganga Aarti"),
    ("Temples", "enable", "Temple Visits"),
    ("Ganga River", "enable", "Boat Rides"),

    # Level 3
    ("Ganga Aarti", "requires", "Evening Visit"),
    ("Ganga Aarti", "type", "Spiritual Activity"),
    ("Boat Rides", "requires", "Early Morning Visit"),

    # Level 4
    ("Evening Visit", "enhances", "Spiritual Experience"),
    ("Early Morning Visit", "enhances", "Sunrise Views"),

    # Level 5
    ("Spiritual Experience", "leads_to", "Inner Peace"),

    # ==================================================================
    #  6. RISHIKESH — Yoga & Adventure Chain
    # ==================================================================
    ("Rishikesh", "has", "Yoga Ashrams"),
    ("Rishikesh", "has", "Rapids"),
    ("Rishikesh", "has", "Suspension Bridges"),
    ("Rishikesh", "located_in", "North India"),
    ("Rishikesh", "best_season", "Monsoon"),
    ("Rishikesh", "budget_level", "Low Budget"),
    ("Rishikesh", "type", "Spiritual Destination"),
    ("Rishikesh", "type", "Adventure Destination"),
    ("Rishikesh", "also_known_as", "Yoga Capital of the World"),

    # Level 2
    ("Yoga Ashrams", "enable", "Yoga Retreats"),
    ("Rapids", "enable", "White Water Rafting"),
    ("Suspension Bridges", "enable", "Bungee Jumping"),

    # Level 3
    ("Yoga Retreats", "requires", "Minimum 7 Days Stay"),
    ("Yoga Retreats", "type", "Wellness Activity"),
    ("White Water Rafting", "requires", "Swimming Skills"),
    ("White Water Rafting", "type", "Adventure Activity"),
    ("Bungee Jumping", "requires", "No Heart Conditions"),

    # Level 4
    ("No Heart Conditions", "checked_via", "Medical Certificate"),

    # Level 5
    ("Medical Certificate", "obtained_from", "Certified Doctor"),

    # ==================================================================
    #  7. LADAKH — High-Altitude & Scenic Chain
    # ==================================================================
    ("Ladakh", "has", "High Altitude Passes"),
    ("Ladakh", "has", "Monasteries"),
    ("Ladakh", "has", "Pangong Lake"),
    ("Ladakh", "located_in", "North India"),
    ("Ladakh", "best_season", "Summer"),
    ("Ladakh", "budget_level", "High Budget"),
    ("Ladakh", "type", "Adventure Destination"),
    ("Ladakh", "requires", "Inner Line Permit"),

    # Level 2
    ("High Altitude Passes", "enable", "Motorcycle Touring"),
    ("Monasteries", "enable", "Cultural Exploration"),
    ("Pangong Lake", "enable", "Photography"),

    # Level 3
    ("Motorcycle Touring", "requires", "High Fitness"),
    ("Motorcycle Touring", "type", "Adventure Activity"),
    ("Cultural Exploration", "requires", "Respectful Behavior"),

    # Level 4
    ("Inner Line Permit", "obtained_from", "District Administration"),
    ("Respectful Behavior", "enhances", "Local Interaction"),

    # Level 5
    ("Local Interaction", "leads_to", "Authentic Experience"),

    # ==================================================================
    #  8. UDAIPUR — Royal Heritage Chain
    # ==================================================================
    ("Udaipur", "has", "Lakes"),
    ("Udaipur", "has", "Palaces"),
    ("Udaipur", "has", "Havelis"),
    ("Udaipur", "located_in", "West India"),
    ("Udaipur", "best_season", "Winter"),
    ("Udaipur", "budget_level", "Moderate Budget"),
    ("Udaipur", "type", "Cultural Destination"),
    ("Udaipur", "also_known_as", "City of Lakes"),
    ("Udaipur", "known_for", "Rajasthani Cuisine"),

    # Level 2
    ("Lakes", "enable", "Boat Rides"),
    ("Havelis", "enable", "Heritage Walks"),

    # Level 3  (Boat Rides & Heritage Walks already defined above)

    # ==================================================================
    #  9. DARJEELING — Tea & Hill Station Chain
    # ==================================================================
    ("Darjeeling", "has", "Tea Gardens"),
    ("Darjeeling", "has", "Toy Train"),
    ("Darjeeling", "has", "Tiger Hill"),
    ("Darjeeling", "located_in", "East India"),
    ("Darjeeling", "best_season", "Summer"),
    ("Darjeeling", "budget_level", "Moderate Budget"),
    ("Darjeeling", "type", "Hill Station"),
    ("Darjeeling", "known_for", "Darjeeling Tea"),

    # Level 2
    ("Tea Gardens", "enable", "Plantation Tours"),
    ("Toy Train", "enable", "Scenic Train Ride"),
    ("Tiger Hill", "enable", "Sunrise Views"),

    # Level 3
    ("Scenic Train Ride", "requires", "Advance Booking"),
    ("Scenic Train Ride", "type", "Cultural Activity"),

    # ==================================================================
    #  10. AGRA — Monument & History Chain
    # ==================================================================
    ("Agra", "has", "Taj Mahal"),
    ("Agra", "has", "Agra Fort"),
    ("Agra", "located_in", "North India"),
    ("Agra", "best_season", "Winter"),
    ("Agra", "budget_level", "Low Budget"),
    ("Agra", "type", "Heritage Destination"),
    ("Agra", "known_for", "Mughlai Cuisine"),

    # Level 2
    ("Taj Mahal", "type", "UNESCO World Heritage Site"),
    ("Taj Mahal", "enable", "Photography"),
    ("Agra Fort", "enable", "History Tours"),

    # Level 3
    ("UNESCO World Heritage Site", "attracts", "International Tourists"),

    # Level 4
    ("International Tourists", "boost", "Local Economy"),

    # Level 5
    ("Local Economy", "supports", "Artisan Crafts"),

    # ==================================================================
    #  11. SHIMLA — Colonial Hill Station Chain
    # ==================================================================
    ("Shimla", "has", "Mall Road"),
    ("Shimla", "has", "Snow"),
    ("Shimla", "has", "Colonial Architecture"),
    ("Shimla", "located_in", "North India"),
    ("Shimla", "best_season", "Summer"),
    ("Shimla", "budget_level", "Moderate Budget"),
    ("Shimla", "type", "Hill Station"),

    # Level 2
    ("Mall Road", "enable", "Shopping"),
    ("Colonial Architecture", "enable", "Heritage Walks"),

    # ==================================================================
    #  12. HAMPI — Ancient Ruins Chain
    # ==================================================================
    ("Hampi", "has", "Ancient Ruins"),
    ("Hampi", "has", "Boulder Landscape"),
    ("Hampi", "located_in", "South India"),
    ("Hampi", "best_season", "Winter"),
    ("Hampi", "budget_level", "Low Budget"),
    ("Hampi", "type", "Heritage Destination"),

    # Level 2
    ("Ancient Ruins", "enable", "Archaeological Exploration"),
    ("Ancient Ruins", "type", "UNESCO World Heritage Site"),
    ("Boulder Landscape", "enable", "Rock Climbing"),

    # Level 3
    ("Archaeological Exploration", "requires", "Guide Booking"),
    ("Archaeological Exploration", "type", "Cultural Activity"),
    ("Rock Climbing", "requires", "High Fitness"),
    ("Rock Climbing", "type", "Adventure Activity"),

    # ==================================================================
    #  13. ANDAMAN ISLANDS — Tropical & Marine Chain
    # ==================================================================
    ("Andaman Islands", "has", "Coral Reefs"),
    ("Andaman Islands", "has", "Beaches"),
    ("Andaman Islands", "has", "Cellular Jail"),
    ("Andaman Islands", "located_in", "East India"),
    ("Andaman Islands", "best_season", "Winter"),
    ("Andaman Islands", "budget_level", "High Budget"),
    ("Andaman Islands", "type", "Leisure Destination"),

    # Level 2
    ("Coral Reefs", "enable", "Scuba Diving"),
    ("Cellular Jail", "enable", "History Tours"),

    # Level 3
    ("Scuba Diving", "requires", "Diving Certification"),
    ("Scuba Diving", "type", "Adventure Activity"),

    # Level 4
    ("Diving Certification", "obtained_from", "PADI Center"),

    # Level 5
    ("PADI Center", "available_at", "Havelock Island"),

    # ==================================================================
    #  14. MUNNAR — Tea & Nature Chain
    # ==================================================================
    ("Munnar", "has", "Tea Plantations"),
    ("Munnar", "has", "Misty Hills"),
    ("Munnar", "has", "Eravikulam National Park"),
    ("Munnar", "located_in", "South India"),
    ("Munnar", "best_season", "Winter"),
    ("Munnar", "budget_level", "Moderate Budget"),
    ("Munnar", "type", "Hill Station"),
    ("Munnar", "type", "Wellness Destination"),

    # Level 2
    ("Misty Hills", "enable", "Trekking"),
    ("Eravikulam National Park", "enable", "Wildlife Safari"),

    # ==================================================================
    #  15. RANN OF KUTCH — Desert & Festival Chain
    # ==================================================================
    ("Rann of Kutch", "has", "White Salt Desert"),
    ("Rann of Kutch", "has", "Rann Utsav Festival"),
    ("Rann of Kutch", "located_in", "West India"),
    ("Rann of Kutch", "best_season", "Winter"),
    ("Rann of Kutch", "budget_level", "Moderate Budget"),
    ("Rann of Kutch", "type", "Cultural Destination"),

    # Level 2
    ("White Salt Desert", "enable", "Desert Safari"),
    ("Rann Utsav Festival", "enable", "Cultural Performances"),

    # Level 3
    ("Desert Safari", "requires", "4x4 Vehicle"),
    ("Desert Safari", "type", "Adventure Activity"),
    ("Cultural Performances", "type", "Cultural Activity"),

    # Level 4
    ("4x4 Vehicle", "available_at", "Local Rental Shops"),

    # ==================================================================
    #  16. COORG — Coffee & Nature Chain
    # ==================================================================
    ("Coorg", "has", "Coffee Plantations"),
    ("Coorg", "has", "Waterfalls"),
    ("Coorg", "has", "Trekking Trails"),
    ("Coorg", "located_in", "South India"),
    ("Coorg", "best_season", "Monsoon"),
    ("Coorg", "budget_level", "Moderate Budget"),
    ("Coorg", "type", "Hill Station"),
    ("Coorg", "also_known_as", "Scotland of India"),

    # Level 2
    ("Coffee Plantations", "enable", "Plantation Tours"),
    ("Waterfalls", "enable", "Photography"),
    ("Trekking Trails", "enable", "Trekking"),

    # ==================================================================
    #  17. OOTY — Nilgiri Hills Chain
    # ==================================================================
    ("Ooty", "has", "Botanical Gardens"),
    ("Ooty", "has", "Nilgiri Mountain Railway"),
    ("Ooty", "has", "Ooty Lake"),
    ("Ooty", "located_in", "South India"),
    ("Ooty", "best_season", "Summer"),
    ("Ooty", "budget_level", "Moderate Budget"),
    ("Ooty", "type", "Hill Station"),
    ("Ooty", "also_known_as", "Queen of Hill Stations"),

    # Level 2
    ("Botanical Gardens", "enable", "Nature Walks"),
    ("Nilgiri Mountain Railway", "enable", "Scenic Train Ride"),
    ("Ooty Lake", "enable", "Boat Rides"),

    # Level 3
    ("Nature Walks", "requires", "Comfortable Footwear"),
    ("Nature Walks", "type", "Wellness Activity"),

    # ==================================================================
    #  18. MYSORE — Royal & Cultural Chain
    # ==================================================================
    ("Mysore", "has", "Mysore Palace"),
    ("Mysore", "has", "Chamundi Hills"),
    ("Mysore", "has", "Brindavan Gardens"),
    ("Mysore", "located_in", "South India"),
    ("Mysore", "best_season", "Winter"),
    ("Mysore", "budget_level", "Low Budget"),
    ("Mysore", "type", "Cultural Destination"),
    ("Mysore", "known_for", "Mysore Cuisine"),
    ("Mysore", "also_known_as", "City of Palaces"),

    # Level 2
    ("Mysore Palace", "enable", "History Tours"),
    ("Chamundi Hills", "enable", "Trekking"),
    ("Brindavan Gardens", "enable", "Musical Fountain Show"),

    # Level 3
    ("Musical Fountain Show", "requires", "Evening Visit"),

    # ==================================================================
    #  19. AMRITSAR — Spiritual & Historical Chain
    # ==================================================================
    ("Amritsar", "has", "Golden Temple"),
    ("Amritsar", "has", "Jallianwala Bagh"),
    ("Amritsar", "has", "Wagah Border"),
    ("Amritsar", "located_in", "North India"),
    ("Amritsar", "best_season", "Winter"),
    ("Amritsar", "budget_level", "Low Budget"),
    ("Amritsar", "type", "Spiritual Destination"),
    ("Amritsar", "known_for", "Punjabi Cuisine"),

    # Level 2
    ("Golden Temple", "enable", "Temple Visits"),
    ("Golden Temple", "type", "Spiritual Landmark"),
    ("Jallianwala Bagh", "enable", "History Tours"),
    ("Wagah Border", "enable", "Border Ceremony"),

    # Level 3
    ("Border Ceremony", "requires", "Afternoon Visit"),
    ("Border Ceremony", "type", "Cultural Activity"),
    ("Spiritual Landmark", "attracts", "International Tourists"),

    # Level 4
    ("Afternoon Visit", "enhances", "Patriotic Experience"),

    # Level 5
    ("Patriotic Experience", "leads_to", "National Pride"),

    # ==================================================================
    #  20. JODHPUR — Desert Fortress Chain
    # ==================================================================
    ("Jodhpur", "has", "Mehrangarh Fort"),
    ("Jodhpur", "has", "Blue Houses"),
    ("Jodhpur", "has", "Desert Landscape"),
    ("Jodhpur", "located_in", "West India"),
    ("Jodhpur", "best_season", "Winter"),
    ("Jodhpur", "budget_level", "Low Budget"),
    ("Jodhpur", "type", "Heritage Destination"),
    ("Jodhpur", "known_for", "Rajasthani Cuisine"),
    ("Jodhpur", "also_known_as", "Blue City"),

    # Level 2
    ("Mehrangarh Fort", "enable", "History Tours"),
    ("Blue Houses", "enable", "Photography"),
    ("Desert Landscape", "enable", "Desert Safari"),

    # Level 3
    ("Mehrangarh Fort", "type", "Architectural Marvel"),

    # Level 4
    ("Architectural Marvel", "attracts", "International Tourists"),


    # ==================================================================
    #  CROSS-CUTTING / SHARED CONCEPT LINKS
    # ==================================================================
    # These connect shared nodes across multiple destinations, making the
    # graph richly interlinked and enabling cross-destination inference.

    # --- Season metadata ---
    ("Winter", "months", "October to February"),
    ("Summer", "months", "March to June"),
    ("Monsoon", "months", "July to September"),

    # --- Budget metadata ---
    ("Low Budget", "daily_cost", "Under 1500 INR per day"),
    ("Moderate Budget", "daily_cost", "1500 to 4000 INR per day"),
    ("High Budget", "daily_cost", "Above 4000 INR per day"),

    # --- Activity-to-category links ---
    ("Adventure Activity", "appeals_to", "Young Travelers"),
    ("Adventure Activity", "requires", "Travel Insurance"),
    ("Cultural Activity", "appeals_to", "History Enthusiasts"),
    ("Wellness Activity", "appeals_to", "Stress Relief Seekers"),
    ("Spiritual Activity", "appeals_to", "Pilgrims"),

    # --- Traveler profile chains ---
    ("Young Travelers", "prefer", "Adventure Destination"),
    ("History Enthusiasts", "prefer", "Heritage Destination"),
    ("Stress Relief Seekers", "prefer", "Wellness Destination"),
    ("Pilgrims", "prefer", "Spiritual Destination"),
    ("Families", "prefer", "Leisure Destination"),
    ("Families", "prefer", "Hill Station"),

    # --- Cuisine cross-links ---
    ("Seafood Cuisine", "type", "Non-Vegetarian"),
    ("Rajasthani Cuisine", "type", "Vegetarian"),
    ("Kerala Cuisine", "type", "Non-Vegetarian"),
    ("Mughlai Cuisine", "type", "Non-Vegetarian"),
    ("Punjabi Cuisine", "type", "Vegetarian"),
    ("Mysore Cuisine", "type", "Vegetarian"),
    ("Street Food", "type", "Vegetarian"),

    # --- Region metadata ---
    ("North India", "climate", "Extreme Seasons"),
    ("South India", "climate", "Tropical"),
    ("West India", "climate", "Dry and Arid"),
    ("East India", "climate", "Humid"),

    # --- Safety / practical chains ---
    ("Travel Insurance", "obtained_from", "Insurance Provider"),
    ("Insurance Provider", "available_at", "Online Portals"),
]


# ──────────────────────────────────────────────────────────────────────
#  QUICK STATISTICS  (computed at import time for debugging/logging)
# ──────────────────────────────────────────────────────────────────────

TOTAL_TRIPLES: int = len(RELATIONSHIPS)
UNIQUE_NODES: set[str] = {
    node
    for triple in RELATIONSHIPS
    for node in (triple[0], triple[2])
}
UNIQUE_PREDICATES: set[str] = {triple[1] for triple in RELATIONSHIPS}

if __name__ == "__main__":
    print("=" * 60)
    print("  Smart India Travel Guide — Semantic Network Stats")
    print("=" * 60)
    print(f"  Total triples      : {TOTAL_TRIPLES}")
    print(f"  Unique nodes       : {len(UNIQUE_NODES)}")
    print(f"  Unique predicates  : {len(UNIQUE_PREDICATES)}")
    print("=" * 60)
    print("\n  Sample 5-level chain (Manali):")
    print("    Manali ─has→ Mountains ─enable→ Trekking")
    print("    ─requires→ High Fitness ─improves→ Better Health")
    print("    ─enables→ Longer Travel")
    print("=" * 60)
