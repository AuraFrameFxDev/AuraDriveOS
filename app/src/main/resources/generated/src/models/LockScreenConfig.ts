/* tslint:disable */
/* eslint-disable */
/**
 * AuraFrameFX Ecosystem API
 * A comprehensive API for interacting with the AuraFrameFX AI Super Dimensional Ecosystem. Provides access to generative AI capabilities, system customization, user management, and core application features. 
 *
 * The version of the OpenAPI document: 1.0.0
 * Contact: support@auraframefx.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { mapValues } from '../runtime';
import type { LockScreenConfigClockConfig } from './LockScreenConfigClockConfig';
import {
    LockScreenConfigClockConfigFromJSON,
    LockScreenConfigClockConfigFromJSONTyped,
    LockScreenConfigClockConfigToJSON,
} from './LockScreenConfigClockConfig';
import type { LockScreenConfigHapticFeedback } from './LockScreenConfigHapticFeedback';
import {
    LockScreenConfigHapticFeedbackFromJSON,
    LockScreenConfigHapticFeedbackFromJSONTyped,
    LockScreenConfigHapticFeedbackToJSON,
} from './LockScreenConfigHapticFeedback';
import type { LockScreenConfigAnimation } from './LockScreenConfigAnimation';
import {
    LockScreenConfigAnimationFromJSON,
    LockScreenConfigAnimationFromJSONTyped,
    LockScreenConfigAnimationToJSON,
} from './LockScreenConfigAnimation';

/**
 * 
 * @export
 * @interface LockScreenConfig
 */
export interface LockScreenConfig {
    /**
     * 
     * @type {LockScreenConfigClockConfig}
     * @memberof LockScreenConfig
     */
    clockConfig?: LockScreenConfigClockConfig;
    /**
     * 
     * @type {LockScreenConfigAnimation}
     * @memberof LockScreenConfig
     */
    animation?: LockScreenConfigAnimation;
    /**
     * 
     * @type {LockScreenConfigHapticFeedback}
     * @memberof LockScreenConfig
     */
    hapticFeedback?: LockScreenConfigHapticFeedback;
}

/**
 * Check if a given object implements the LockScreenConfig interface.
 */
export function instanceOfLockScreenConfig(value: object): value is LockScreenConfig {
    return true;
}

export function LockScreenConfigFromJSON(json: any): LockScreenConfig {
    return LockScreenConfigFromJSONTyped(json, false);
}

export function LockScreenConfigFromJSONTyped(json: any, ignoreDiscriminator: boolean): LockScreenConfig {
    if (json == null) {
        return json;
    }
    return {
        
        'clockConfig': json['clockConfig'] == null ? undefined : LockScreenConfigClockConfigFromJSON(json['clockConfig']),
        'animation': json['animation'] == null ? undefined : LockScreenConfigAnimationFromJSON(json['animation']),
        'hapticFeedback': json['hapticFeedback'] == null ? undefined : LockScreenConfigHapticFeedbackFromJSON(json['hapticFeedback']),
    };
}

export function LockScreenConfigToJSON(value?: LockScreenConfig | null): any {
    if (value == null) {
        return value;
    }
    return {
        
        'clockConfig': LockScreenConfigClockConfigToJSON(value['clockConfig']),
        'animation': LockScreenConfigAnimationToJSON(value['animation']),
        'hapticFeedback': LockScreenConfigHapticFeedbackToJSON(value['hapticFeedback']),
    };
}

