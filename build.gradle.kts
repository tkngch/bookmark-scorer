import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.4.21"
    kotlin("plugin.serialization") version "1.4.10"

    id("org.jlleitschuh.gradle.ktlint") version "9.4.0"
    id("com.palantir.git-version") version "0.12.3"

    id("maven-publish")
}

group = "tkngch"
// https://github.com/palantir/gradle-git-version/issues/188
val gitVersion: groovy.lang.Closure<String> by extra
version = gitVersion()

repositories {
    mavenCentral()
    jcenter() // for pytorch
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.0.1")
    implementation("org.pytorch:pytorch_java_only:1.7.1")
    testImplementation(kotlin("test-junit"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.4.2")
}

tasks.test {
    // note that libtorch needs to be downloaded from here:
    // https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip
    systemProperty("java.library.path", file("libtorch/lib").absolutePath)

    useJUnitPlatform()
    testLogging.events = setOf(
        org.gradle.api.tasks.testing.logging.TestLogEvent.PASSED,
        org.gradle.api.tasks.testing.logging.TestLogEvent.FAILED,
        org.gradle.api.tasks.testing.logging.TestLogEvent.SKIPPED,
        org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_OUT,
        org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_ERROR
    )
    testLogging.showCauses = true
    testLogging.showExceptions = true
    testLogging.showStackTraces = true
    testLogging.exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}

ktlint {
    verbose.set(true)
    outputToConsole.set(true)
    enableExperimentalRules.set(true)
    filter {
        exclude { element -> element.file.path.contains("generated/") }
    }
}

publishing {
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/tkngch/bookmark-scorer")
            credentials {
                username = System.getenv("GITHUB_ACTOR")
                password = System.getenv("GITHUB_TOKEN")
            }
        }
    }
    publications {
        create<MavenPublication>("gpr") {
            from(components["java"])
        }
    }
}
