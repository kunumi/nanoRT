# tuples are (database, table, target column, leakage columns)

forecast_clf_tasks = [
    ("rel-amazon", "user-churn", "churn", []),
    ("rel-hm", "user-churn", "churn", []),
    ("rel-stack", "user-badge", "WillGetBadge", []),
    ("rel-amazon", "item-churn", "churn", []),
    ("rel-stack", "user-engagement", "contribution", []),
    ("rel-avito", "user-visits", "num_click", []),
    ("rel-avito", "user-clicks", "num_click", []),
    ("rel-event", "user-ignore", "target", []),
    ("rel-trial", "study-outcome", "outcome", []),
    ("rel-f1", "driver-dnf", "did_not_finish", []),
    ("rel-event", "user-repeat", "target", []),
    ("rel-f1", "driver-top3", "qualifying", []),
]

autocomplete_clf_tasks = [
    ("rel-avito", "SearchInfo", "IsUserLoggedOn", []),
    ("rel-stack", "postLinks", "LinkTypeId", []),
    ("rel-amazon", "review", "verified", []),
    ("rel-trial", "studies", "has_dmc", []),
    (
        "rel-trial",
        "eligibilities",
        "adult",
        [
            "child",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    (
        "rel-trial",
        "eligibilities",
        "child",
        [
            "adult",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    ("rel-event", "event_interest", "not_interested", ["interested"]),
]
