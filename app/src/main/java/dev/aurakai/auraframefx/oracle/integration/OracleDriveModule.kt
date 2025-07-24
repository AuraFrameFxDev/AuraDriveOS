package dev.aurakai.auraframefx.oracledrive

import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * OracleDrive Dependency Injection Module
 * Integrates Oracle Drive services into AuraFrameFX ecosystem
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class OracleDriveModule {
    
    /**
     * Binds the OracleDriveServiceImpl implementation to the OracleDriveService interface as a singleton.
     *
     * This allows dependency injection to provide a singleton instance of OracleDriveServiceImpl wherever OracleDriveService is required.
     *
     * @param oracleDriveServiceImpl The implementation of OracleDriveService to bind.
     * @return The bound OracleDriveService instance.
     */
    @Binds
    @Singleton
    abstract fun bindOracleDriveService(
        oracleDriveServiceImpl: OracleDriveServiceImpl
    ): OracleDriveService
}