from functools import total_ordering


@total_ordering
class MaterializedView:
    def __init__(self, name, definition_sql, estimated_size=None, dependent_indexes=None):
        if not name or not definition_sql:
            raise ValueError("MaterializedView needs a name and definition SQL")
        self.name = name
        self.definition_sql = definition_sql
        # Store estimated size when available
        self.estimated_size = estimated_size
        # Optional indexes that can be created on this MV
        self.dependent_indexes = tuple(dependent_indexes or [])
        # For tracking in database (similar to hypopg_name for indexes)
        self.db_oid = None

    # Used to sort materialized views
    def __lt__(self, other):
        if not isinstance(other, MaterializedView):
            return False
        return self.name < other.name

    def __repr__(self):
        return f"MV({self.name})"

    def __eq__(self, other):
        if not isinstance(other, MaterializedView):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def table_name(self):
        """Return the materialized view name as table name for index operations"""
        return self.name

    def get_dependent_indexes(self):
        """Get all dependent indexes for this materialized view"""
        return self.dependent_indexes

    def add_dependent_index(self, index):
        """Add a dependent index to this materialized view"""
        if index not in self.dependent_indexes:
            self.dependent_indexes = self.dependent_indexes + (index,)

    def is_equivalent_to(self, other):
        """Check if two materialized views are equivalent (same definition)"""
        if not isinstance(other, MaterializedView):
            return False
        return (self.name == other.name and 
                self.definition_sql.strip().lower() == other.definition_sql.strip().lower())

    def get_size_estimate(self):
        """Get estimated size in bytes"""
        return self.estimated_size or 0

    def set_size_estimate(self, size_bytes):
        """Set estimated size in bytes"""
        self.estimated_size = size_bytes
