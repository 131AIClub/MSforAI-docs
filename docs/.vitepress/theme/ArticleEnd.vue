<script setup lang="ts">
import ContributorsList from './ContributorsList.vue'
import CoursePagination from './CoursePagination.vue'
import GiscusComments from './GiscusComments.vue'
</script>

<template>
  <div class="article-end">
    <CoursePagination />
    <ContributorsList />
    <GiscusComments />
  </div>
</template>
