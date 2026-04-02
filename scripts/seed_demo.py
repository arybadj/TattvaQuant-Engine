from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.base import Base
from app.models import Company, Job, User
from app.models.common import RoleEnum
from app.utils.security import hash_password

settings = get_settings()


def main() -> None:
    engine = create_engine(settings.sync_database_url, future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
      company = session.execute(select(Company).where(Company.slug == "acme")).scalar_one_or_none()
      if not company:
          company = Company(name="Acme Corp", slug="acme", timezone="UTC")
          session.add(company)
          session.flush()

      if not session.execute(select(User).where(User.email == "admin@acme.example")).scalar_one_or_none():
          session.add_all(
              [
                  User(
                      company_id=company.id,
                      full_name="Admin User",
                      email="admin@acme.example",
                      hashed_password=hash_password("Password123!"),
                      role=RoleEnum.admin,
                      timezone="UTC",
                  ),
                  User(
                      company_id=company.id,
                      full_name="Interviewer One",
                      email="interviewer@acme.example",
                      hashed_password=hash_password("Password123!"),
                      role=RoleEnum.interviewer,
                      timezone="UTC",
                  ),
              ]
          )

      if not session.execute(select(Job).where(Job.company_id == company.id, Job.title == "Senior Backend Engineer")).scalar_one_or_none():
          session.add(
              Job(
                  company_id=company.id,
                  title="Senior Backend Engineer",
                  department="Engineering",
                  location="Remote",
                  description=(
                      "Design distributed systems, build Python microservices, and own production reliability. "
                      "Experience with APIs, PostgreSQL, Redis, queues, cloud deployment, and AI integrations is preferred."
                  ),
                  requirements="Python, FastAPI, PostgreSQL, Redis, Docker, distributed systems, APIs",
              )
          )

      session.commit()
      print("Demo seed complete.")


if __name__ == "__main__":
    main()

