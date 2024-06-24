from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider



def get_cassandra_session():
    cluster = Cluster()
    session = cluster.connect('shares')
    return session, cluster


if __name__ == "__main__":
    session, cluster = get_cassandra_session()
    session.shutdown()
    cluster.shutdown()