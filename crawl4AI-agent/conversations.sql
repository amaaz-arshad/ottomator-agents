-- Enable the uuid-ossp extension if you want server-side UUID defaults
-- (in Supabase you can also pass UUID from the client, but this shows one option)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE public.conversations (
    id         uuid         PRIMARY KEY DEFAULT uuid_generate_v4(),
    title      text         NOT NULL,
    messages   jsonb        NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz  NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at timestamptz  NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Automatically keep updated_at in sync on any UPDATE
CREATE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = timezone('utc'::text, now());
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_conversations_updated_at
BEFORE UPDATE ON public.conversations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
