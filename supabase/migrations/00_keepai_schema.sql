-- =============================================================================
-- KEEP.AI  -  SQL Migration (единый файл)
-- =============================================================================
-- Базовая схема для текстового ИИ-ассистента.
-- Объединяет прежние файлы 00_models_01…00_models_06 в одну миграцию.
-- Порядок применения: сверху вниз.
-- =============================================================================

-- ═════════════════════════════════════════════════════════════════════════════
-- 0.  РАСШИРЕНИЯ
-- ═════════════════════════════════════════════════════════════════════════════
create extension if not exists vector with schema extensions;

-- ═════════════════════════════════════════════════════════════════════════════
-- 1.  ПРОФИЛИ
-- ═════════════════════════════════════════════════════════════════════════════
create table if not exists public.profiles (
    id              uuid            primary key    references auth.users(id) on delete cascade,
    full_name       text            not null default '',
    avatar_url      text            not null default '',
    role            text            not null default 'user' check (role in ('user', 'admin')),
    created_at      timestamptz     not null default now(),
    updated_at      timestamptz     not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
as $$
begin
    insert into public.profiles (id, full_name, avatar_url, role)
    values (new.id, new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'avatar_url', 'user');
    return new;
end;
$$;

create trigger on_auth_user_created after insert on auth.users
    for each row execute procedure public.handle_new_user();

alter table public.profiles enable row level security;
create policy "profiles: read own" on public.profiles for select using (auth.uid() = id);
create policy "profiles: insert own" on public.profiles for insert with check (auth.uid() = id);
create policy "profiles: update own" on public.profiles for update using (auth.uid() = id);
create index idx_profiles_user_role on public.profiles (role);

-- ═════════════════════════════════════════════════════════════════════════════
-- 2.  ГАЛЕРЕЯ ВЕРСИЙ (AI-моделей)
-- ═════════════════════════════════════════════════════════════════════════════
create table if not exists public.versions (
    id              uuid            primary key    default gen_random_uuid(),
    user_id         uuid            not null       references public.profiles(id) on delete cascade,
    label           text            not null,
    version         text            not null default '',
    model_name      text            not null default '',
    provider        text            not null default 'openai',
    is_enabled      boolean         not null default true,
    temperature     numeric         not null default 0.7,
    max_tokens      integer         not null default 2048,
    created_at      timestamptz     not null default now(),
    updated_at      timestamptz     not null default now()
);

alter table public.versions enable row level security;
create policy "versions: read own" on public.versions for select using (auth.uid() = user_id);
create policy "versions: insert own" on public.versions for insert with check (auth.uid() = user_id);
create policy "versions: update own" on public.versions for update using (auth.uid() = user_id);
create policy "versions: delete own" on public.versions for delete using (auth.uid() = user_id);
create index idx_versions_user_enabled on public.versions (user_id, is_enabled);
create index idx_versions_label on public.versions (label);

create or replace function public.update_updated_at_column()
returns trigger
language plpgsql
as $$
begin new.updated_at = now(); return new; end;
$$;

create trigger trg_versions_updated_at before update on public.versions
    for each row execute function public.update_updated_at_column();

-- ═════════════════════════════════════════════════════════════════════════════
-- 3.  ДИАЛОГИ (CONVERSATIONS)
-- ═════════════════════════════════════════════════════════════════════════════
create table if not exists public.conversations (
    id              uuid            primary key    default gen_random_uuid(),
    user_id         uuid            not null       references public.profiles(id) on delete cascade,
    title           text            not null default 'New chat',
    version_id      uuid            not null       references public.versions(id) on delete cascade,
    status          text            not null default 'active' check (status in ('active', 'archived', 'deleted')),
    created_at      timestamptz     not null default now(),
    updated_at      timestamptz     not null default now()
);

alter table public.conversations enable row level security;
create policy "conversations: read own" on public.conversations for select using (auth.uid() = user_id);
create policy "conversations: insert own" on public.conversations for insert with check (auth.uid() = user_id);
create policy "conversations: update own" on public.conversations for update using (auth.uid() = user_id);
create policy "conversations: delete own" on public.conversations for delete using (auth.uid() = user_id);
create index idx_conversations_user on public.conversations (user_id, updated_at desc);
create index idx_conversations_version on public.conversations (version_id);

create trigger trg_conversations_updated_at before update on public.conversations
    for each row execute function public.update_updated_at_column();

-- ═════════════════════════════════════════════════════════════════════════════
-- 4.  СООБЩЕНИЯ (MESSAGES)
-- ═════════════════════════════════════════════════════════════════════════════
create table if not exists public.messages (
    id              uuid            primary key    default gen_random_uuid(),
    conversation_id uuid            not null       references public.conversations(id) on delete cascade,
    user_id         uuid            not null       references public.profiles(id) on delete cascade,
    role            text            not null       check (role in ('user', 'assistant', 'system')),
    content         text            not null,
    embedding       vector(1536),
    tokens_used     integer         not null default 0,
    model_id        text            not null default '',
    created_at      timestamptz     not null default now()
);

alter table public.messages enable row level security;
create policy "messages: read conversation" on public.messages for select
    using (exists (select 1 from public.conversations c where c.id = messages.conversation_id and c.user_id = auth.uid()));
create policy "messages: insert user" on public.messages for insert
    with check (auth.uid() = user_id and exists (select 1 from public.conversations c where c.id = messages.conversation_id and c.user_id = auth.uid()));
create policy "messages: insert assistant" on public.messages for insert
    with check (true);
create index idx_messages_conversation on public.messages (conversation_id, created_at);
create index idx_messages_user on public.messages (user_id, created_at);

-- ═════════════════════════════════════════════════════════════════════════════
-- 5.  EМБЕДДИНГИ (VECTOR) / КОНТЕКСТЫ
-- ═════════════════════════════════════════════════════════════════════════════
create table if not exists public.knowledge_documents (
    id              uuid            primary key    default gen_random_uuid(),
    user_id         uuid            not null       references public.profiles(id) on delete cascade,
    source_file     text            not null,
    title           text            not null,
    section_title   text            default '',
    content         text            not null,
    content_hash    text            not null unique,
    embedding       vector(1536)    default null,
    metadata        jsonb           not null default '{}',
    created_at      timestamptz     not null default now()
);

alter table public.knowledge_documents enable row level security;
create policy "knowledge: read all" on public.knowledge_documents for select using (true);
create policy "knowledge: insert own" on public.knowledge_documents for insert with check (auth.uid() = user_id);
create policy "knowledge: update own" on public.knowledge_documents for update using (auth.uid() = user_id);
create policy "knowledge: delete own" on public.knowledge_documents for delete using (auth.uid() = user_id);
create index idx_knowledge_hash on public.knowledge_documents (content_hash);
create index idx_knowledge_user on public.knowledge_documents (user_id);

create or replace function public.match_knowledge_documents(
    query_embedding      extensions.vector,
    match_count          integer         default 5,
    match_threshold      double precision default 0.7,
    user_id_filter       uuid            default null
)
returns table (
    id                 uuid,
    source_file        text,
    title              text,
    section_title      text,
    content            text,
    metadata           jsonb,
    similarity         double precision
)
language plpgsql
security definer
set search_path = public, extensions
as $$
begin
    return query
    select kd.id, kd.source_file, kd.title, kd.section_title, kd.content, kd.metadata,
           1 - (kd.embedding <=> query_embedding)::double precision as similarity
    from public.knowledge_documents kd
    where 1 - (kd.embedding <=> query_embedding) >= match_threshold
      and ($4 is null or kd.user_id = $4)
    order by kd.embedding <=> query_embedding
    limit $2;
end;
$$;

grant execute on function public.match_knowledge_documents(extensions.vector, integer, double precision, uuid) to authenticated;

-- ═════════════════════════════════════════════════════════════════════════════
-- 6.  РЕАКЦИИ НА СООБЩЕНИЯ
-- ═════════════════════════════════════════════════════════════════════════════
create table if not exists public.message_reactions (
    id              uuid            primary key    default gen_random_uuid(),
    message_id      uuid            not null       references public.messages(id) on delete cascade,
    user_id         uuid            not null       references public.profiles(id) on delete cascade,
    reaction_type   text            not null default 'like' check (reaction_type in ('like', 'dislike', 'love', 'surprised', 'angry', 'sad')),
    created_at      timestamptz     not null default now()
);

alter table public.message_reactions enable row level security;
create policy "reactions: read" on public.message_reactions for select
    using (exists (select 1 from public.messages m where m.id = message_reactions.message_id and m.user_id = auth.uid()));
create policy "reactions: insert own" on public.message_reactions for insert with check (auth.uid() = user_id);
create policy "reactions: delete own" on public.message_reactions for delete using (auth.uid() = user_id);
create index idx_reactions_message on public.message_reactions (message_id);
create index idx_reactions_user on public.message_reactions (user_id);

-- ═════════════════════════════════════════════════════════════════════════════
-- 7.  ПРЕДСТАВЛЕНИЯ ДЛЯ ФРОНТЕНДА
-- ═════════════════════════════════════════════════════════════════════════════
create or replace view public.user_conversations AS
select c.id as conversation_id, c.title as conversation_title, c.status as conversation_status,
       c.updated_at as conversation_updated_at, c.version_id,
       (select m.content from public.messages m where m.conversation_id = c.id order by m.created_at desc limit 1) as last_message,
       (select m.role from public.messages m where m.conversation_id = c.id order by m.created_at desc limit 1) as last_message_role,
       count(m.id) as message_count
from public.conversations c
left join public.messages m on m.conversation_id = c.id
where c.user_id = auth.uid() and c.status != 'deleted'
group by c.id, c.title, c.status, c.updated_at, c.version_id
order by c.updated_at desc;

-- ═════════════════════════════════════════════════════════════════════════════
-- 8.  ПРАВА ДОСТУПА
-- ═════════════════════════════════════════════════════════════════════════════
grant select, insert, update, delete on all tables in schema public to authenticated;
grant execute on all functions in schema public to authenticated;
alter default privileges in schema public grant select, insert, update, delete on tables to authenticated;
alter default privileges in schema public grant execute on functions to authenticated;

-- ═════════════════════════════════════════════════════════════════════════════
-- 9.  СТАРТОВОЕ ЗАПОЛНЕНИЕ
-- ═════════════════════════════════════════════════════════════════════════════
-- Одна запись 'AI Model 1' на каждый профиль (если ещё нет).
-- Примечание: user_id обязан ссылаться на существующий профиль (FK),
-- поэтому сид привязывается к profiles, а не к случайному uuid.
insert into public.versions (user_id, label, version, model_name, provider, is_enabled, temperature, max_tokens)
select p.id, 'AI Model 1', '1.0', 'gpt-4o-mini', 'openai', true, 0.7, 2048
from public.profiles p
where not exists (
    select 1 from public.versions v
    where v.user_id = p.id and v.label = 'AI Model 1'
);

-- ═════════════════════════════════════════════════════════════════════════════
-- ГОТОВО!  Импорт: Supabase > Database > Migrations > Import SQL
-- (файл применяется сверху вниз одной миграцией)
-- =============================================================================